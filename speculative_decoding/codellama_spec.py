# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import time
from datasets import load_from_disk
import numpy as np
import json
import random
import os
import tabulate as tab
from collections import defaultdict
from typing import Iterable, Dict
import gzip
import logging

class LlamaModelEval(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs)
        en = time.perf_counter()
        logging.critical(f"[PROFILE] model_decoder_forward {en-st}")
        return outputs


class LlamaModelEval_Draft(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.model_name = None

    def forward(self, *args, **kwargs):
        st = time.perf_counter()
        outputs = super().forward(*args, **kwargs) 
        en = time.perf_counter()
        logging.critical(f"[PROFILE] draft_model_decoder_forward {en-st}")
        codellama_extra_token_number = 16
        outputs['logits'] = torch.nn.functional.pad(outputs['logits'], (0, codellama_extra_token_number), value=float('-inf'))
        return outputs



class ProfileLLM:
    start_idx_arr = []
    end_idx_arr = []
    generate_times_arr = []
    num_tokens_out_arr = []
    num_tokens_in_arr = []
    exec_time_arr = []
    prompt_time_arr = []
    token_time_arr = []
    decoder_time_arr = []

    @classmethod
    def clear_entries(cls):
        cls.start_idx_arr = []
        cls.end_idx_arr = []
        cls.generate_times_arr = []
        cls.num_tokens_out_arr = []
        cls.num_tokens_in_arr = []
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []

    @classmethod
    def collect_sections(cls, logf):
        ### Default is not using AG
        ag = False
        for i, line in enumerate(logf):
            line = line.lstrip().rstrip().split(" ")
            if len(line)>1:
                #print(f"line: {line}")
                if line[1] == "tokenizer:":
                    cls.start_idx_arr.append(i + 1)
                elif line[1] == "generate:":
                #     import pdb
                #     pdb.set_trace()
                    cls.end_idx_arr.append(i)
                    t = float(line[2])
                    cls.generate_times_arr.append(t)
                    num_tokens_out = int(line[4])
                    cls.num_tokens_out_arr.append(num_tokens_out)
                    num_tokens_in = int(line[7].rstrip(";"))
                    cls.num_tokens_in_arr.append(num_tokens_in)
                if "draft_model_decoder_forward" in line:
                    ag = True
        print(f"\n\nNumber of prompts found in log: {len(cls.start_idx_arr)}\n")
        return ag

    @classmethod
    def parse_section(
        cls,
        outlog,
        filename,
        prompt_num,
        logf,
        start_idx,
        end_idx,
        generate_time,
        num_tokens_out,
        num_tokens_in
        ):
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []
        cls.decoder_time_arr_draft = []

        for i in range(start_idx, end_idx, 1):
            line = logf[i].lstrip().rstrip().split(" ")
            # print(f"line : {line}")
            if line[1] != "model_decoder_forward":
                #print(f"line : {line}")
                m = int(line[1])
                k = int(line[2])
                n = int(line[4])
                exec_time_start = float(line[11])
                exec_time_end = float(line[12])
                exec_time = (exec_time_end - exec_time_start)
                cls.exec_time_arr.append(exec_time)
                if m > 1:
                    cls.prompt_time_arr.append(exec_time)
                else:
                    cls.token_time_arr.append(exec_time)
            elif line[1] == "model_decoder_forward":
                #print(f"line : {line}")
                decoder_time = float(line[2])
                cls.decoder_time_arr.append(decoder_time)
            else:
                decoder_time = float(line[2])
                cls.decoder_time_arr_draft.append(decoder_time)

        matmul_prompt_time = sum(cls.prompt_time_arr)
        matmul_token_time = sum(cls.token_time_arr)
        matmul_cumulative_time = sum(cls.exec_time_arr)
        other_layers_time = (generate_time - matmul_cumulative_time )
        # import pdb
        # pdb.set_trace()
        new_tokens_generated = num_tokens_out - num_tokens_in

        if len(cls.decoder_time_arr) > 0:
            decoder_time_prefill_phase = cls.decoder_time_arr[0]
            # decoder_time_token_phase = sum(cls.decoder_time_arr[1:])
            decoder_time_token_phase = generate_time - decoder_time_prefill_phase
            prefill_phase = decoder_time_prefill_phase * 1e3
            if new_tokens_generated > 1:
                time_per_token = (decoder_time_token_phase *1e3)/ (new_tokens_generated - 1)
                # tokens_per_sec = 1000.0 / time_per_token
                tokens_per_sec = new_tokens_generated / generate_time 
            else:
                time_per_token = "na"
                tokens_per_sec = "na"
        else:
            decoder_time_prefill_phase = "na"
            decoder_time_token_phase = "na"
            prefill_phase = "na"
            time_per_token = "na"
            tokens_per_sec = "na"

        outlog.write(
            f"{filename},{prompt_num+1},{num_tokens_in},{matmul_prompt_time:.3f},{matmul_token_time:.3f},{matmul_cumulative_time:.3f},{other_layers_time:.3f},{generate_time:.3f},{decoder_time_prefill_phase},{decoder_time_token_phase},{num_tokens_out},{new_tokens_generated},{prefill_phase},{time_per_token},{tokens_per_sec}\n")

        outlog.flush()

        return [
            prompt_num + 1,
            num_tokens_in,
            new_tokens_generated,
            generate_time ,
            prefill_phase,
            time_per_token,
            tokens_per_sec,
            1,
            1,
            1
        ]

    

    @classmethod
    def parse_section_ag(
        cls,
        outlog,
        filename,
        prompt_num,
        logf,
        start_idx,
        end_idx,
        generate_time,
        num_tokens_out,
        num_tokens_in
        ):
        cls.exec_time_arr = []
        cls.prompt_time_arr = []
        cls.token_time_arr = []
        cls.decoder_time_arr = []
        cls.decoder_time_arr_draft = []
        cls.valid_tokens = []

        for i in range(start_idx, end_idx, 1):
            line = logf[i].lstrip().rstrip().split(" ")
            # print(f"line : {line}")
            if line[1] != "model_decoder_forward" and line[1] != "draft_model_decoder_forward" and line[1] != "valid_tokens":
                m = int(line[1])
                k = int(line[2])
                n = int(line[4])
                exec_time_start = float(line[11])
                exec_time_end = float(line[12])
                exec_time = (exec_time_end - exec_time_start)
                cls.exec_time_arr.append(exec_time)
                if m > 1:
                    cls.prompt_time_arr.append(exec_time)
                else:
                    cls.token_time_arr.append(exec_time)
            elif line[1] == "valid_tokens":
                valid_tokens = float(line[2])
                cls.valid_tokens.append(valid_tokens)
            elif line[1] == "model_decoder_forward":
                decoder_time = float(line[2])
                cls.decoder_time_arr.append(decoder_time)
            else:
                decoder_time = float(line[2])
                cls.decoder_time_arr_draft.append(decoder_time)
        # import pdb
        # pdb.set_trace()
        matmul_prompt_time = sum(cls.prompt_time_arr)
        matmul_token_time = sum(cls.token_time_arr)
        matmul_cumulative_time = sum(cls.exec_time_arr)
        other_layers_time = (generate_time - matmul_cumulative_time )
        # import pdb
        # pdb.set_trace()
        new_tokens_generated = num_tokens_out - num_tokens_in

        if len(cls.decoder_time_arr) > 0:
            # decoder_time_prefill_phase = cls.decoder_time_arr[0]
            # decoder_time_token_phase = sum(cls.decoder_time_arr[1:])
            decoder_time_prefill_phase = cls.decoder_time_arr[0] + sum(cls.decoder_time_arr_draft[:5])
            # decoder_time_token_phase = sum(cls.decoder_time_arr[1:]) + sum(cls.decoder_time_arr_draft[5:])
            decoder_time_token_phase = generate_time - decoder_time_prefill_phase

            accepted_num = sum(cls.valid_tokens) - len(cls.decoder_time_arr)
            guessed_num = len(cls.decoder_time_arr_draft)

            prefill_phase = decoder_time_prefill_phase * 1e3
            if new_tokens_generated > 1:
                # time_per_token = (decoder_time_token_phase *1e3)/ (new_tokens_generated - 1)
                time_per_token = (decoder_time_token_phase *1e3)/ (new_tokens_generated - cls.valid_tokens[0])
                tokens_per_sec = new_tokens_generated / generate_time 
            else:
                time_per_token = "na"
                tokens_per_sec = "na"
        else:
            decoder_time_prefill_phase = "na"
            decoder_time_token_phase = "na"
            prefill_phase = "na"
            time_per_token = "na"
            tokens_per_sec = "na"
            accept_rate = "na"

        outlog.write(
            f"{filename},{prompt_num+1},{num_tokens_in},{matmul_prompt_time:.3f},{matmul_token_time:.3f},{matmul_cumulative_time:.3f},{other_layers_time:.3f},{generate_time:.3f},{decoder_time_prefill_phase},{decoder_time_token_phase},{num_tokens_out},{new_tokens_generated},{prefill_phase},{time_per_token},{tokens_per_sec}\n")

        outlog.flush()

        return [
            prompt_num + 1,
            num_tokens_in,
            new_tokens_generated,
            generate_time ,
            prefill_phase,
            time_per_token,
            tokens_per_sec,
            accepted_num,
            guessed_num,
            cls.valid_tokens[0]
        ]

    @classmethod
    def analyze_profiling(cls, in_file, out_file):
        out_file.write(
            "Filename,Example#,Num_Tokens_In,MatMul_time_Prefill_phase[s],MatMul_time_Token_phase[s],MatMul_time_Cumulative[s],All_Other_layers[s],Generate_Time[s],Decoder_time_Prefill_phase[s],Decoder_time_Token_phase[s],Num_Tokens_Out,Num_New_Tokens,Prefill_Phase[ms],Time_per_Token[ms],Tokens\sec\n"
        )
        with open(in_file, "r") as f:
            logf = f.readlines()
            ag = cls.collect_sections(logf)

            perf_table = [
                [
                    "Example#",
                    "Prompt Length (tokens)",
                    "New Tokens Generated",
                    "Total Time (s)",
                    "Prefill Phase (ms)",
                    "Time/Token (ms)",
                    "Tokens/Sec",
                    "Accept num",
                    "Guess num",
                    "New Tokens Generated at first step", 
                ]
            ]
            if ag:
                for i in range(len(cls.start_idx_arr)):
                    perf_table.append(
                        cls.parse_section_ag(
                            out_file,
                            in_file,
                            i,
                            logf,
                            cls.start_idx_arr[i],
                            cls.end_idx_arr[i],
                            cls.generate_times_arr[i],
                            cls.num_tokens_out_arr[i],
                            cls.num_tokens_in_arr[i],
                        )
                    )
            else:
                for i in range(len(cls.start_idx_arr)):
                    perf_table.append(
                        cls.parse_section(
                            out_file,
                            in_file,
                            i,
                            logf,
                            cls.start_idx_arr[i],
                            cls.end_idx_arr[i],
                            cls.generate_times_arr[i],
                            cls.num_tokens_out_arr[i],
                            cls.num_tokens_in_arr[i],
                        )
                    )
            print(tab.tabulate(perf_table, headers="firstrow", tablefmt="github"))
        inference_Time_model_generate = 0
        Prefill_time = 0
        generated_token_number = 0
        generated_token_number_at_first_step = 0
        max_generated_token_number = -1
        min_generated_token_number = 10000000
        Decoding_Latency = 0
        Troughput = 0
        accept_num = 0
        guess_num = 0
        for pt in perf_table[1:]:
            pt = [float(elem) for elem in pt]
            inference_Time_model_generate += pt[3]
            Prefill_time += pt[4]/1e3
            generated_token_number += pt[2]
            max_generated_token_number = pt[2] if max_generated_token_number < pt[2] else max_generated_token_number
            min_generated_token_number = pt[2] if min_generated_token_number > pt[2] else min_generated_token_number
            generated_token_number_at_first_step += pt[-1]
            accept_num += pt[-3]
            guess_num += pt[-2]
        Decoding_Latency = (inference_Time_model_generate-Prefill_time) / (generated_token_number-generated_token_number_at_first_step)
        Troughput = (generated_token_number/inference_Time_model_generate)
        print("inference_Time_model_generate: ", inference_Time_model_generate)
        print("Prefill_time: ", Prefill_time)
        print('generated_token_number: ', generated_token_number)
        print("max_generated_token_number: ", max_generated_token_number)
        print("min_generated_token_number: ", min_generated_token_number)
        print("Decoding_Latency: ", Decoding_Latency)
        print("Troughput: ", Troughput)
        print("acceptance rate: ", accept_num/guess_num)

#-------------------------begin evaluation------------------------------
def benchmark_code(model, log_file, max_seqlen=512, assistant_model=None, sample=False, temperature=None, stopping=None):
    """put all the relevant function into this benchmark function"""
    def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
        """
        Writes an iterable of dictionaries to jsonl
        """
        if append:
            mode = 'ab'
        else:
            mode = 'wb'
        filename = os.path.expanduser(filename)
        if filename.endswith(".gz"):
            with open(filename, mode) as fp:
                with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                    for x in data:
                        gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
        else:
            with open(filename, mode) as fp:
                for x in data:
                    fp.write((json.dumps(x) + "\n").encode('utf-8'))

    def clip_input(tokenizer, prompt, max_new_tokens=512, max_seql=4096):
        system_prompt = "# python code to complete some task. # Create a function to calculate the sum of a sequence of integers. [PYTHON]\ndef sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum \n[/PYTHON]\n#"
        prompt = prompt['prompt']
        prompt = system_prompt + prompt
        input_ids = tokenizer(prompt,return_tensors='pt').input_ids
        if len(input_ids[0])+max_new_tokens>=max_seql:
            print('(input ids+max token)> {}'.format(max_seql))
            sample_num = (len(input_ids[0])+max_new_tokens-max_seql) 
            input_ids = torch.cat((input_ids[0][:2],input_ids[0][2:-3][:-sample_num],input_ids[0][-3:]),dim=0).unsqueeze(0)
        return  input_ids

    # humaneval data
    def get_humaneval(path=None):
        if path != None:
            with open(path, 'r') as json_file:
                humaneval_data = json.load(json_file)
            return humaneval_data
        else:
            from datasets import load_dataset
            prompt_data = load_dataset("openai_humaneval")
            return prompt_data

    def count_indent(text: str) -> int:
        count = 0
        for char in text:
            if char == " ":
                count += 1
            else:
                break
        return count


    def fix_indents(text: str, multiple: int = 2):
        outputs = []
        for line in text.split("\n"):
            while count_indent(line) % multiple != 0:
                line = " " + line
            outputs.append(line)
        return "\n".join(outputs)


    def filter_code(completion: str, model=None) -> str:
        completion = completion.lstrip("\n")
        return completion.split("\n\n")[0]
    

    testloader = get_humaneval()["test"]

    task_name = "humaneval"
    big_reorg_dict = defaultdict(list)

    for i, prompt in enumerate(testloader):
        task_id = prompt["task_id"]
        input_ids = clip_input(model.tokenizer, prompt).to(model.device)
        logging.critical(f"[PROFILE] tokenizer:")

        start = time.perf_counter()
        generate_ids = model.generate(input_ids, do_sample=sample, max_new_tokens=max_seqlen, pad_token_id=model.tokenizer.eos_token_id,
                temperature=temperature,
                stopping_criteria=stopping,
                top_k=10, top_p=0.95,
                assistant_model=assistant_model)
        end = time.perf_counter()
        generate_time = (end - start)
        prompt_tokens = input_ids.shape[1]
        num_tokens_out = generate_ids.shape[1]
        new_tokens_generated = num_tokens_out - prompt_tokens
        time_per_token = (generate_time/new_tokens_generated)*1e3
        logging.critical(f"[PROFILE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}")
        completion = model.tokenizer.decode(generate_ids[0, input_ids.shape[1] : ])
        completion = filter_code(fix_indents(completion))
        val_completion = completion
        print(f"response: {val_completion}")
        logging.critical(f"response: {val_completion}")

        #-------------------------below is results store-------------------------------------
        comp_item = dict(task_id="{}".format(task_id), completion=val_completion)
        big_reorg_dict["yxl"].append(comp_item)

    
    result_data_path = './results/codellama_spec_{}_on_mi250'.format(task_name)
    if not os.path.exists(result_data_path):
        os.makedirs(result_data_path)

    for key, completion in big_reorg_dict.items():
        write_jsonl("{0}/{1}.jsonl".format(result_data_path, key), completion)

    # similar to other benchmark methods
    logging.shutdown()
    out_file = log_file.replace(".log", "_profile.csv")
    out_file = open(out_file, "w")
    ProfileLLM.analyze_profiling(log_file, out_file)
    out_file.close()


def warmup(model, prompts, max_new_tokens=30):
    print(f"Warming up ... ")
    for prompt in prompts[0:1]:
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
        generate_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens)
        _ = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Warm up DONE!! ")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="target_model", help="Target model path")
    parser.add_argument("--draft_model", type=str, default=None, help="Draft model path")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens")
    parser.add_argument("--do_sample", action='store_true', help="Whether to use sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproduction")
    parser.add_argument("--device", type=str, default="cuda", help="Device for models")
    parser.add_argument("--bf16", action='store_true', help="dtype for models")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    codellama_checkpoint = args.target_model
    assistant_checkpoint = args.draft_model

    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(codellama_checkpoint)
    model = LlamaModelEval.from_pretrained(codellama_checkpoint, torch_dtype=torch_dtype).to(device)
    model.tokenizer = tokenizer

    sample = args.do_sample
    use_spec = args.draft_model != None
    temperature = args.temperature
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if use_spec:
        log_file = log_dir + "/log_codellama_spec.log"
    else:
        log_file = log_dir + "/log_codellama_target.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        level=logging.CRITICAL)


    warmup_prompts = ["from typing import List\n\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each of the group, output the deepest level of nesting of parentheses.\n    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"\n"]
    warmup(model, warmup_prompts)
    if use_spec:
        assistant_model = LlamaModelEval_Draft.from_pretrained(assistant_checkpoint, torch_dtype=torch_dtype).to(device)
        benchmark_code(model, log_file, assistant_model=assistant_model, sample=sample, temperature=temperature)
    else:
        benchmark_code(model, log_file, sample=sample, temperature=temperature)


if __name__ == "__main__":
    main()
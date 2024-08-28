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


export HF_DATASETS_CACHE="./huggingface_data"

for task in wikitext lambada_openai winogrande piqa sciq wsc arc_easy arc_challenge logiqa hellaswag mmlu truthfulqa gsm8k ceval-valid
do
    export CUDA_VISIBLE_DEVICES="0"
    lm_eval --model hf \
        --tasks $task \
        --model_args pretrained=/path/to/your/huggingface/model \
        --device cuda:0 \
        --batch_size 2 
done

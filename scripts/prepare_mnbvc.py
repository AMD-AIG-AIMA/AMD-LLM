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


import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

import pandas as pd
import gzip

def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_mnbvc_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        try:
            # contents = pd.read_parquet(filepath, engine='pyarrow')['content']
            if 'code/metadata/' in filepath:
                print("Not use metadata!")
                continue
            with gzip.open(open(filepath, "rb"), mode="rt") as f:
                for row in tqdm(f):
                    text = json.loads(row)["text"]
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
        except:
            print(f"Error reading {filepath}!!")
            continue

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 4097 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
) -> None:
    import time
    assert split == "train" #  starcoder only has train data
    filenames = glob.glob(os.path.join(source_path, "*/*/*.jsonl.gz"), recursive=True)
    filenames += glob.glob(os.path.join(source_path, "*/*/*/*.jsonl.gz"), recursive=True)
    print(len(filenames))
    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]
    filenames = filenames[:int(len(filenames) * percentage)]
    num_processes = 64
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)

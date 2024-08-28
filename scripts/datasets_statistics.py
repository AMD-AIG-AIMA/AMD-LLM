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


import glob
import os

data_path = '../slim_processed'
chunk_size=4097*1024

prefix = 'train_*'
data_split = {
    'starcoder': 'train_starcoder',
    'slimpajama_wiki': 'train_wikipedia_slimpajama',
    'slimpajama_git': 'train_github_slimpajama',
    'slimpajama_book': 'train_book_slimpajama',
    'slimpajama': 'train_slimpajama',
    'mnbvc': 'train_mnbvc',
    'skypile': 'train_skypile',
    'openwebmath': 'train_openwebmath',
    'project_gutenberg': 'train_project_gutenberg',
}
data_epoches = {
    'starcoder': 1.0,
    'slimpajama_wiki': 1.0,
    'slimpajama_git': 1.0,
    'slimpajama_book': 1.0,
    'slimpajama': 1.0,
    'mnbvc': 1.0,
    'skypile': 1.0,
    'openwebmath': 1.0,
    'project_gutenberg': 1.0,
}
data_statis = {}
for data_name in data_split:
    data_statis[data_name] = 0
total_chunks = 0
total_tokens = 0

filenames = glob.glob(os.path.join(data_path, prefix), recursive=True)
for filename in filenames:
    for data_name, pref in data_split.items():
        if filename[len(os.path.dirname(filename))+1:].startswith(pref):
            data_statis[data_name] += 1
            total_chunks += 1
print('statistics:')
for data_name, num_chunk in data_statis.items():
    print(f'{num_chunk*chunk_size/1000/1000/1000} B tokens, ', f'{num_chunk} chunks, ', data_name)
    total_tokens += num_chunk*chunk_size
print(f"{total_tokens/1000/1000/1000} B tokens", f"{total_chunks} chunks in total.")

print("percentage:")
for data_name, num_chunk in data_statis.items():
    print(f'1.0 epoches, {num_chunk*chunk_size / total_tokens} %, ', data_name)

print("weighted:")
total_tokens = 0
for data_name, num_chunk in data_statis.items():
    print(f'{num_chunk*chunk_size*data_epoches[data_name]/1000/1000/1000} B tokens, ', f'{num_chunk} chunks, ', data_name)
    total_tokens += num_chunk*chunk_size*data_epoches[data_name]

for data_name, num_chunk in data_statis.items():
    print(f'{data_epoches[data_name]} epoches, {num_chunk*chunk_size*data_epoches[data_name] / total_tokens*100} %, ', data_name)
print(f"{total_tokens/1000/1000/1000} B tokens", f"{total_chunks} chunks in total.")

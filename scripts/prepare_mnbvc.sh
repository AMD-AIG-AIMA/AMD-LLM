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


source_path=../MNBVC
target_path=../mnbvc_processed/
tokenizer_path=./scripts/tokenizer

python ./scripts/prepare_mnbvc.py \
    --source_path $source_path \
    --tokenizer_path $tokenizer_path  \
    --destination_path $target_path \
    --split train \
    --percentage 1.0

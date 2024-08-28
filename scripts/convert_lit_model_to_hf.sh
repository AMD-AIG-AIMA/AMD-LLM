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


iter=104000
model_path=./out_bak_100m_2k_code_iter_$iter/
mkdir $model_path
checkpoint_name=iter-${iter}-ckpt.pth
cp ./out/tinyllama_135M_2k/$checkpoint_name $model_path
python scripts/convert_lit_checkpoint.py \
    --checkpoint_name=$checkpoint_name\
    --out_dir=$model_path \
    --model_name='tiny_LLaMA_135M_2k' \
    --model_only=False

cp ./scripts/tokenizer/* ${model_path}
mv ${model_path}/iter-${iter}-ckpt.bin ${model_path}/pytorch_model.bin
rm ${model_path}/${checkpoint_name}

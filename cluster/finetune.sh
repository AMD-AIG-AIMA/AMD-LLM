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


export TRAIN_DATA_PATH=/path/to/preprocessed/training/data
export VALID_DATA_PATH=/path/to/preprocessed/validation/data
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export BASE_MODEL_PATH=/path/to/base/model/ckpt
export NCCL_SOCKET_IFNAME={network interface name}
export MASTER_ADDRESS={master node ip}
export MAIN_OPRT={port}

MODEL_NAME='tiny_LLaMA_135M_2k'
lightning run model \
    --node-rank=0  \
    --main-address=$MASTER_ADDRESS \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    --main-port=$MAIN_OPRT \
    pretrain/tinyllama_code.py --devices 8 --train_data_dir $TRAIN_DATA_PATH  --val_data_dir $VALID_DATA_PATH --model_name $MODEL_NAME \
    --checkpoint_path $BASE_MODEL_PATH

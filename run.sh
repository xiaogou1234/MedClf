#!/bin/bash

# 后台运行脚本命令
# cd MedClf
# nohup bash ./run.sh > ./log/run.log 2>&1 &

# 数据集名称
datasets=("retinamnist" "dermamnist" "bloodmnist")

# 模型名称
# models=("next_vit")
models=("resnet" "vit" "convnext" "vgg" "swin_transformer" "next_vit")

# 学习率
lr=1e-4

# 批处理大小
batch_size=32

# 最大迭代次数
max_iter=30

# 循环运行所有数据集和模型组合
for dataset in "${datasets[@]}"; do
    mkdir -p "results/${dataset}"
    for model in "${models[@]}"; do
        echo "Running on dataset: ${dataset} with model: ${model}"
        python main.py --dataset_name "${dataset}" --model_name "${model}" --lr "${lr}" --batch_size "${batch_size}" --max_iter "${max_iter}"
    done
done

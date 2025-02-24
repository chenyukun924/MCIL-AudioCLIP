#!bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name classroom33_3-6-MoE-Adapters.yaml \
    dataset_root="data" \
    class_order="class_orders/classroom_data.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name classroom33_5-5-MoE-Adapters.yaml \
    dataset_root="data" \
    class_order="class_orders/classroom_data.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name imagenet19_3-6.yaml \
    dataset_root="data" \
    class_order="class_orders/imagenet19.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name imagenet19_6-3.yaml \
    dataset_root="data" \
    class_order="class_orders/imagenet19.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name imagenet27_5-5.yaml \
    dataset_root="data" \
    class_order="class_orders/imagenet27.yaml"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name imagenet27_8-3.yaml \
    dataset_root="data" \
    class_order="class_orders/imagenet27.yaml"
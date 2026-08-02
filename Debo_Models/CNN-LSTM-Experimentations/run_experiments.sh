#!/bin/bash

DATA_PATH="/media/nacho/Station_s Vault/DATA/2026/Persistence-Cell-Migration-Project/DATA/"

MODELS=("cnn" "lstm")
TASKS=("dr_only" "h_only" "joint")
SPLITS=("60_30_10" "70_20_10" "80_15_5")

for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        for task in "${TASKS[@]}"; do
            echo "===================================================="
            echo " RUNNING: Model=$model | Split=$split | Task=$task"
            echo "===================================================="
            
            # Train the individual model
            #python3 train_pipeline.py \
                #--data_path "$DATA_PATH" \
                #--model_type "$model" \
                #--task_name "$task" \
                #--split_name "$split" \
                #--epochs 45 \
                #--batch_size 64 \
                #--lr 0.0005

            # Evaluate the individual model
            python3 evaluate_pipeline.py \
                --data_path "$DATA_PATH" \
                --model_type "$model" \
                --task_name "$task" \
                --split_name "$split" \
                --batch_size 64
                
        done
    done
done
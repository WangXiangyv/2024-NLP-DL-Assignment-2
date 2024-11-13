#!/bin/bash
# A bash script for running all the fine-parameter fine-tuning experiments  (45 in total)
# The script will typically call proxychains4 api for accessing Internet. If needn't, just delete "proxychains4 -q" prefix in line 13

for model in roberta-base
do
    for dataset in restaurant_sup acl_sup agnews_sup
    do
        for seed in {1..5}
        do
            echo "$model $dataset $seed $peft"
            proxychains4 -q python train.py\
                --cache_dir model\
                --do_train 1\
                --do_eval 1\
                --output_dir output\
                --overwrite_output_dir 1\
                --wandb_project NLP-DL-A2\
                --report_to wandb\
                --run_name "$model-$dataset-$seed-lora"\
                --logging_steps 1\
                --eval_strategy epoch\
                --seed $seed\
                --learning_rate 5e-5\
                --num_train_epochs 10\
                --per_device_train_batch_size 32\
                --per_device_eval_batch_size 32\
                --dataset_name $dataset\
                --model_name_or_path $model
        done
    done
done
#!/bin/bash

for model in bert-base-uncased roberta-base allenai/scibert_scivocab_uncased
do
    for dataset in restaurant_sup acl_sup agnews_sup
    do
        for seed in {1..5}
        do
            lr=5e-5
            echo "$model $dataset $seed $peft $lr"
            proxychains4 -q python train.py\
                --cache_dir model\
                --do_train 1\
                --do_eval 1\
                --output_dir output\
                --overwrite_output_dir 1\
                --wandb_project NLP-DL-A2\
                --report_to wandb\
                --run_name "$model-$dataset-$seed-$peft"\
                --logging_steps 1\
                --eval_strategy epoch\
                --seed $seed\
                --learning_rate $lr\
                --num_train_epochs 10\
                --per_device_train_batch_size 32\
                --per_device_eval_batch_size 32\
                --peft 0\
                --dataset_name $dataset\
                --model_name_or_path $model
        done
    done
done
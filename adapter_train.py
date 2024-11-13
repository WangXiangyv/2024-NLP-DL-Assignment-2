import logging
import os
import sys
from dataHelper import get_dataset, implemented_datasets
import datasets
import transformers
from transformers import (
    TrainingArguments,
    set_seed,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    AutoTokenizer
)
import numpy as np
import evaluate
import wandb
from RoBERTa_Adapter import RobertaForSeqClsWithAdapter, RobertaWithAdapterConfig, RobertaLayerWithAdapter

cache_dir = "adapter_model"
run_name = "roberta-adapter"

#hyparameters
dataset_name = "agnews_sup"
adapter_act = "gelu"
adapter_bottleneck = 32
seed = 2024
lr = 1e-4
epoch = 6

# Set up logger
logger = logging.getLogger(__name__)

def main():
    '''
    Initialize logging and wandb
    '''
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    os.environ["WANDB_PROJECT"] = "RobertaWithAdapter"  # name your W&B project
    os.environ["WANDB_CACHE_DIR"] = "wandb"
    set_seed(seed)
    transformers.utils.logging.set_verbosity_info()
    log_level = "INFO"
    logger.setLevel(log_level)
    
    '''
    Initialize Dataset and Model
    '''
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", cache_dir=cache_dir)
    # Load dataset with tokenizer's sep-token
    raw_dataset = get_dataset(dataset_name, sep_token=tokenizer.sep_token)
    # Load model config
    model_config = RobertaWithAdapterConfig.from_pretrained(
        "FacebookAI/roberta-base", cache_dir=cache_dir, num_labels=len(raw_dataset['train'].unique('label'))
    )
    model_config.adapter_act = adapter_act
    model_config.adapter_bottleneck_dim = adapter_bottleneck
    # Load model
    model = RobertaForSeqClsWithAdapter.from_pretrained("FacebookAI/roberta-base", cache_dir=cache_dir, config=model_config)
    
    '''
    Process datasets and build up dataloader
    '''
    # Determine max_seq_length
    max_seq_length = tokenizer.model_max_length
    # Define preprocess function for map
    def preprocess_func(examples):
        result = tokenizer(examples['text'], padding=False, max_length=max_seq_length, truncation=True)
        result['label'] = examples['label']
        return result
    # Map to processed dataset
    raw_dataset = raw_dataset.map(
        preprocess_func,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    # Get train dataset and eval dataset
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["test"]
    # Setup dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    '''
    Prepare Metrics
    '''
    f1_metric = evaluate.load("f1", cache_dir=cache_dir)
    acc_metric = evaluate.load("accuracy", cache_dir=cache_dir)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        ret = {
            "micro-f1":f1_metric.compute(predictions=preds, references=p.label_ids, average="micro"),
            "macro-f1":f1_metric.compute(predictions=preds, references=p.label_ids, average="macro"),
            "accuracy":acc_metric.compute(predictions=preds, references=p.label_ids)
        }
        # ret = f1_metric.compute(predictions=preds, reference=p.label_ids, average="micro")
        return ret
    
    '''
    Initialize trainer
    '''
    #Set training args
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        num_train_epochs=epoch,
        output_dir="RobertaWithAdapter",
        overwrite_output_dir=True,
        report_to="wandb",
        run_name = '-'.join([dataset_name, str(lr), str(seed)]),
        logging_steps=10,
        eval_strategy="epoch",
        learning_rate=lr
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    '''
    Training
    '''
    train_result = trainer.train()
    output_metrics = train_result.metrics
    max_train_samples = len(train_dataset)
    output_metrics["train_samples"] = max_train_samples
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", output_metrics)
    trainer.save_metrics("train", output_metrics)
    trainer.save_state()
    wandb.finish()

if __name__ == '__main__':
    main()
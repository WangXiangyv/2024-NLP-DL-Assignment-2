import logging
import os
import sys
import wandb
from dataHelper import get_dataset, implemented_datasets
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer
)
import evaluate
from adapters import AutoAdapterModel, AdapterTrainer, DoubleSeqBnConfig, LoRAConfig


'''
Set up logger and define argument dataclasses
'''
# Set up logger
logger = logging.getLogger(__name__)

# Define argument dataclasses
@dataclass
class DataArguments:
    dataset_name: str|List[str] = field(
        metadata={"help": "The name(s) of the dataset(s) to use."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    def __post_init__(self):
        # Check whether provided dataset name valid
        if self.dataset_name is not None:
            invalid_ds_name = []
            for item in self.dataset_name:
                if item not in implemented_datasets:
                    invalid_ds_name.append(item)
            if len(invalid_ds_name) > 0:
                raise ValueError(f'Invalid dataset name: {','.join(invalid_ds_name)}.')

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to store the pretrained models downloaded from huggingface."}
    )
    peft_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Peft mode to use."}
    )

@dataclass
class WandbArguments:
    connect_wandb: bool = field(
        default=True
    )
    wandb_project: str = field(
        default='untitled_project'
    )
    wandb_cache_dir: str = field(
        default='wandb_cache'
    )

'''
Define main function
'''
def main():
    
    '''
    Initialize argparser, logging, seed, wandb, cuda
    '''
    # Set cuda visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # depend on specific device
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, WandbArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, wandb_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, wandb_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel("INFO")
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Set wandb
    if wandb_args.connect_wandb:
        os.environ["WANDB_PROJECT"] = wandb_args.wandb_project  # name W&B project
        os.environ["WANDB_CACHE_DIR"] = wandb_args.wandb_cache_dir

    '''
    Load model, tokenizer, dataset
    '''
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # Load dataset with tokenizer's sep-token
    raw_dataset = get_dataset(data_args.dataset_name, sep_token=tokenizer.sep_token)
    
    # Load model
    if model_args.peft is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(raw_dataset['train'].unique('label')),
            cache_dir=model_args.cache_dir,
        )
        logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad)) # Report the number of trainable params
        
    else:
        model = AutoAdapterModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        if model_args.peft == "adapter":
            adapter_config = DoubleSeqBnConfig(reduction_factor=16)
        elif model_args.peft == "lora":
            adapter_config = LoRAConfig(r=32)
        else:
            raise ValueError(f"Undefined peft method {model_args.peft}")
        model.add_adapter("PEFT", config=adapter_config)
        model.add_classification_head(
            "PEFT", 
            num_labels=len(raw_dataset['train'].unique('label')), 
            id2label={i:v for i,v in enumerate(raw_dataset['train'].unique('label'))}
        )
        model.train_adapter("PEFT")
        logger.info(model.adapter_summary()) # Report the number of trainable params

    '''
    Process datasets and build up dataloader
    '''
    # Pre-process
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    def preprocess_func(examples):
        result = tokenizer(examples['text'], padding=padding, max_length=max_seq_length, truncation=True)
        return result
    raw_dataset = raw_dataset.map(
        preprocess_func,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["test"]
    
    # Load metrics
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        ret = {
            "micro-f1":f1_metric.compute(predictions=preds, references=p.label_ids, average="micro"),
            "macro-f1":f1_metric.compute(predictions=preds, references=p.label_ids, average="macro"),
            "accuracy":acc_metric.compute(predictions=preds, references=p.label_ids)
        }
        return ret

    # Build up datacollator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    '''
    Training
    '''
    if model_args.peft is None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    if training_args.do_train:
        train_result = trainer.train()
        output_metrics = train_result.metrics

        trainer.save_model()  # Save the tokenizer too for easy upload
        trainer.log_metrics("train", output_metrics)
        trainer.save_metrics("train", output_metrics)
        trainer.save_state()

    if wandb_args.connect_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
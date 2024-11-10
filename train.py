import logging
import os
import sys
from dataHelper import get_dataset, implemented_datasets
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import datasets
import transformers
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
from transformers.trainer_utils import get_last_checkpoint
import evaluate
import wandb

'''
Set up logger and define argument dataclasses
'''
# Set up logger
logger = logging.getLogger(__name__)

# Define argument dataclasses
@dataclass
class DataTrainingArguments:
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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class WandbArguments:
    connect_wandb: bool = field(
        default=True
    )
    wandb_project: str = field(
        default='untitled_project'
    )
    wandb_log_model: str = field(
        default='end'
    )
    wandb_cache_dir: str = field(
        default='wandb cache'
    )

'''
Define main function
'''
def main():
    
    '''
    Initialize argparser, logging, seed, wandb
    '''
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, WandbArguments, TrainingArguments))
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
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    # Set wandb
    if wandb_args.connect_wandb:
        os.environ["WANDB_PROJECT"] = wandb_args.wandb_project  # name your W&B project
        os.environ["WANDB_LOG_MODEL"] = wandb_args.wandb_log_model  # log all model checkpoints
        os.environ["WANDB_CACHE_DIR"] = wandb_args.wandb_cache_dir
    
    '''
    Load model, tokenizer, dataset
    '''
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Load dataset with tokenizer's sep-token
    raw_dataset = get_dataset(data_args.dataset_name, sep_token=tokenizer.sep_token)
    
    # Load model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(raw_dataset['train'].unique('label')),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    '''
    Process datasets and build up dataloader
    '''
    # Whether pad all samples to max_seq_len. If not, will pad dynamically.
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    
    # Determine max_seq_length
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    # Define preprocess function for map
    def preprocess_func(examples):
        result = tokenizer(examples['text'], padding=padding, max_length=max_seq_length, truncation=True)
        result['label'] = examples['label']
        return result
    
    # Map to processed dataset
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_dataset = raw_dataset.map(
            preprocess_func,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        
    # Get train dataset and eval dataset
    train_dataset = raw_dataset["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    eval_dataset = raw_dataset["test"]
    
    # # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    # Load metrics
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)
    acc_metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
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

    # Build up datacollator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    '''
    Initialize trainer
    '''
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
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        output_metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        output_metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", output_metrics)
        trainer.save_metrics("train", output_metrics)
        trainer.save_state()
    if wandb_args.connect_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
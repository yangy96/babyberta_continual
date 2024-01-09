"""
Train a Roberta model using code provided by library maintainers
"""

import logging
import os
import json
from datasets import Dataset, DatasetDict

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments, AutoConfig
import sys

from babyberta.io import load_sentences_from_file
from babyberta.utils import make_sequences, split
from babyberta import configs
from babyberta.params import param2default, Params

import argparse
import torch

os.environ["WANDB_DISABLED"] = "true"

def main(text_path,save_path,read_path, seed, unmasked_removal):

    params = Params.from_param2val(param2default)

    # get new rep
    rep = 0
    path_out = configs.Dirs.root / 'official_implementation' / str(rep)
    while path_out.exists():
        rep += 1
        path_out = configs.Dirs.root / 'official_implementation' / str(rep)

    print(f'replication={rep}')
    saved_training_args = torch.load('training_args.bin')
    
    training_args = TrainingArguments(
        output_dir='results_'+save_path.split('/')[-1],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=saved_training_args.per_device_train_batch_size,
        per_device_eval_batch_size=saved_training_args.per_device_train_batch_size,
        learning_rate=saved_training_args.learning_rate,
        weight_decay=saved_training_args.weight_decay,
        max_steps=saved_training_args.max_steps,  #max steps changed to 26000000
        warmup_steps=saved_training_args.warmup_steps,
        num_train_epochs = params.num_epochs,
        gradient_accumulation_steps=saved_training_args.gradient_accumulation_steps, # 8 for 2 gpu
        seed=seed,
        eval_steps=saved_training_args.save_steps, 
        evaluation_strategy='steps',
        save_steps=saved_training_args.save_steps, 
    )
    
    
    print("args",saved_training_args)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    set_seed(seed)

    logger.info("Loading data")
    data_path = configs.Dirs.corpora / text_path  # we use aonewsela for reference implementation
    sentences = load_sentences_from_file(data_path,
                                         include_punctuation=params.include_punctuation,
                                         allow_discard=True)
    train_sentences, test_sentences = split(sentences)
    data_in_dict = {'text': make_sequences(train_sentences, params.num_sentences_per_input)}
    test_data_in_dict = {'text': make_sequences(test_sentences, params.num_sentences_per_input)}
    datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict),'test': Dataset.from_dict(test_data_in_dict)})
    print(datasets)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    logger.info("Loading tokenizer")
    
    
    print(str(configs.Dirs.tokenizers / params.tokenizer/'.json'))
    if "roberta" in read_path:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        print("roberta")
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(read_path,add_prefix_space=True)

    model = RobertaForMaskedLM.from_pretrained(read_path,)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = "text"

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=128,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    print("length",len(train_dataset))
    #print(f'Length of train data={len(train_dataset)}')

    # Data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                    mlm_probability=0.15, unmasked_removal=unmasked_removal)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    trainer.train() #resume_from_checkpoint=True
    trainer.save_model(save_path)  # Saves the tokenizer too
    #tokenizer.save_pretrained(save_path)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess qamr file into huggingface format')
    parser.add_argument('--text_path',type=str, default="wikipedia1.txt", help='path for reading pretrained model')
    parser.add_argument('--save_path',type=str, default='BabyBERTa-100M', help='whether to train a new question answering model')
    parser.add_argument('--read_path',type=str, default='./BabyBERTa_AO-CHILDES', help='read the pretrained model')
    parser.add_argument('--seed',type=int, default=523, help='a random seed')
    parser.add_argument('--unmasked_removal',type=bool, default=False, help='unmasking removal policy')
    args = parser.parse_args()
    main(args.text_path, args.save_path+'_'+str(args.seed), args.read_path, args.seed,args.unmasked_removal)
    with open(args.save_path+'_'+str(args.seed)+'/'+'training_dataset.json','w') as f:
        json.dump({'text_path':args.text_path,'save_path':args.save_path+'_'+str(args.seed),'read_path':args.read_path,'URP':args.unmasked_removal,'seed':args.seed},f)
    f.close()

    
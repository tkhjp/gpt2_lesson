# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：朱老师, 微信: 15000361164
@version：
@Date：
@Desc: run training for a light GPT model
"""


import copy
import math
import os
import random
from itertools import chain


import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, default_data_collator, get_scheduler, AutoConfig, AutoModelForCausalLM
from transformers.utils.logging import get_logger

import config
from model import GPTConfig, GPT

logger = get_logger(__name__)


################################
# acclerater
accelerator_log_kwargs = {}
accelerator_log_kwargs["log_with"] = "all"
accelerator_log_kwargs["logging_dir"] = config.output_dir
accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    # **accelerator_log_kwargs
)


##################################
# load model, include configs

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# print(tokenizer.vocab)

# 如果是直接用huggingface的模型：
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# print("model.config.eos_token_id: ", model.config.eos_token_id)


if config.init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=len(tokenizer),
        dropout=config.dropout
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
else:
    assert config.init_from.startswith("gpt")
    # resume training from a pretrained model.
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=config.dropout)
    model = GPT.from_pretrained(config.init_from, override_args)

print(model)

os.makedirs(config.output_dir, exist_ok=True)

# load dataset
dataset = load_dataset(
    "text",
    data_files={
        "train": "datasets/仙逆.txt"
    }
)

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['validation'] = split_dataset.pop('test') # rename the test split to val

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = split_dataset["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    list_texts = []
    for exp in examples[text_column_name]:
        # print(exp)
        if exp == "":
            exp = '<|endoftext|>'  # BERT: [CLS], [SEP], [MASK], [unused]；
        list_texts.append(exp)
    res = tokenizer(list_texts)
    return res

with accelerator.main_process_first():
    tokenized_datasets = split_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=config.load_from_cache_file,
        desc="Running tokenizer on dataset",
    )

# 限定的模型接收的最大长度
block_size = min(config.block_size, tokenizer.model_max_length)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    # print(examples.keys())
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # print(total_length)
    # 尾部可能有部分tokens被去除了.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

with accelerator.main_process_first():
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=config.load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=config.batch_size
)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=config.batch_size
)

# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.001,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=config.warmup_iters * config.gradient_accumulation_steps,
    num_training_steps=max_train_steps * config.gradient_accumulation_steps,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)
experiment_config = vars(config)
for key in copy.copy(list(experiment_config.keys())):
    if key.startswith("_"):
        experiment_config.pop(key)

print(experiment_config)
# TensorBoard cannot log Enums, need the raw value
experiment_config["lr_scheduler_type"] = "linear"
accelerator.init_trackers("clm_no_trainer", experiment_config)


#############################
# Training loop
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
for epoch in range(0, config.num_train_epochs):
    model.train()

    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        # print("batch: ", batch.keys())

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        # if completed_steps % config.checkpointing_steps == 0 and completed_steps > 0:
        #     output_dir_ = f"step_{completed_steps }"
        #     if config.output_dir is not None:
        #         output_dir = os.path.join(config.output_dir, output_dir_)
        #     accelerator.save_state(output_dir)
        if completed_steps >= max_train_steps:
            break

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(config.batch_size)))
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

    accelerator.log(
        {
            "perplexity": perplexity,
            "eval_loss": eval_loss,
            "train_loss": total_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        },
        step=completed_steps,
    )

    accelerator.wait_for_everyone()
    output_dir_ = os.path.join(config.output_dir, f"epoch_{epoch}")
    accelerator.save_state(output_dir_)
    tokenizer.save_pretrained(config.output_dir)


'''

accelerate config

accelerate launch main.py

'''

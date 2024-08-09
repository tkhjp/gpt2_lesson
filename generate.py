# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：朱老师, 微信: 15000361164
@version：
@Date：
@Desc: run generate with our code
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

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token


# init the model and laod checkpoints
print("Initializing a new model from scratch")
model_args = dict(
    n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd, block_size=config.block_size,
    bias=config.bias, vocab_size=len(tokenizer), dropout=config.dropout
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# load state dict.
model.load_state_dict(
    torch.load("experiments/outputs/run1/epoch_9/pytorch_model.bin")
)
model.eval()

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# print(model)

device = torch.device("cuda")
model = model.to(device)

# 输入一些 prompt
input_prompt = [
    "Today is a beautiful day, and",
    "good study, day day",
    "王林嘴角带着一丝冷笑,"
]
batch_inputs = tokenizer(
    input_prompt,
    return_tensors="pt",
    padding=True,
    # max_length=128
)

input_ids = batch_inputs["input_ids"]
input_ids = input_ids.to(device)
print(input_ids)

input_tokens = tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=False)
print(input_tokens)



# # top_k_sampling
# generated_sequencess = model.generate(
#     input_ids,
#     max_new_tokens=24,
#     num_beams=5,
#     temperature=1.0,
#     top_k=3,
#     p_nucleus=None,
#     method="top_k_sampling"
# )
# print(generated_sequencess)

# # nucleus_sampling
# generated_sequencess = model.generate(
#     input_ids,
#     max_new_tokens=12,
#     num_beams=5,
#     temperature=1.0,
#     top_k=None,
#     p_nucleus=0.9,
#     method="nucleus_sampling"
# )
# print(generated_sequencess)


# beam_search
generated_sequencess = model.generate(
    input_ids,
    max_new_tokens=12,
    num_beams=5,
    temperature=1.0,
    top_k=None,
    p_nucleus=None,
    method="beam_search"
)
print(generated_sequencess)

generated_sequencess = tokenizer.batch_decode(generated_sequencess, skip_special_tokens=False)
print(generated_sequencess)


'''
python generate.py

'''

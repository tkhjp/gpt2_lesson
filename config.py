# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：朱老师, 微信: 15000361164
@version：
@Date：
@Desc: configs
"""


# -----------------------------------------------------------------------------
# config

# I/O
output_dir = 'experiments/outputs/run1'
eval_interval = 2000
log_interval = 1
eval_iters = 200
do_train = True
do_eval = True
always_save_checkpoint = True
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'wikitxt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
load_from_cache_file = False
dataset = 'wikitext'
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
batch_size = 32   # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256

# model
n_layer = 3
n_head = 8
n_embd = 128
dropout = 0.0 # pretraining 时候 设置dropout 为 0， finetuning 时候设置0.1或者更多
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
num_train_epochs = 10
checkpointing_steps = 2500

# DDP settings
# handled by accelerate


# -----------------------------------------------------------------------------
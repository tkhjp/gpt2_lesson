# -*- coding: utf-8 -*-
"""
@Copyright: 2024 Dr Zhu.
@License：the Apache License, Version 2.0
@Author：朱老师, 微信: 15000361164
@version：
@Date：
@Desc: generation from hf utils
"""

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
# result = generator("good study, day day ", max_length=30, num_return_sequences=5)
result = generator("Today is a beautiful day, and ", max_length=30, num_return_sequences=5)
print(result)
#!/usr/bin/env bash

#export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface
#./hh-train.py
#exit 1

export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface
#python hh-train-parallel.py --language_pairs sr-en --output_dir ./phi-3-medium-LoRA-sr-en --model_id microsoft/Phi-3-medium-4k-instruct
#python hh-train-parallel.py --language_pairs is-en --output_dir ./phi-3-mini-LoRA-is-en --model_id microsoft/Phi-3-mini-4k-instruct
#python hh-train-parallel.py --language_pairs is-en --output_dir ./phi-3-medium-LoRA-is-en --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 30 --max_steps 80
python hh-train-parallel.py --language_pairs sr-en,is-en --output_dir ./phi-3-medium-LoRA-is-en --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 30 --max_steps 80

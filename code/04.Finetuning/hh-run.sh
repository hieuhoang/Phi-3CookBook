#!/usr/bin/env bash

#export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface
#./hh-train.py
#exit 1

export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface
#python hh-train-parallel.py --language_pairs sr-en --output_dir ./phi-3-medium-LoRA-sr-en --model_id microsoft/Phi-3-medium-4k-instruct
#python hh-train-parallel.py --language_pairs is-en --output_dir ./phi-3-mini-LoRA-is-en --model_id microsoft/Phi-3-mini-4k-instruct
#python hh-train-parallel.py --language_pairs is-en --output_dir ./phi-3-medium-LoRA-is-en --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 30 --max_steps 80
# python hh-train-parallel.py --language_pairs sr-en,is-en --output_dir ./phi-3-medium-LoRA-is-en --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 30 --max_steps 80
# exit 0

NAME=medium.de && singsub --name $NAME --gpu G16 --vc lang-sing-wu3 --venv /opt/envs/hihoan/phi.amd/bin/activate python hh-train-parallel.py --language_pairs de-en --output_dir $NAME --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 5000

NAME=medium.gp1 && singsub --name $NAME --gpu G16 --vc lang-sing-wu3 --venv /opt/envs/hihoan/phi.amd/bin/activate python hh-train-parallel.py --language_pairs en-da,en-nl,en-de,en-is,en-no,en-sv,en-af --output_dir $NAME --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 5000

NAME=medium.all && singsub --name $NAME --gpu G16 --vc lang-sing-wu3 --venv /opt/envs/hihoan/phi.amd/bin/activate python hh-train-parallel.py --language_pairs af-en,ar-en,az-en,bg-en,ca-en,zh-en,cs-en,da-en,nl-en,et-en,fi-en,fr-en,gl-en,ka-en,de-en,el-en,gu-en,he-en,hi-en,hu-en,is-en,id-en,it-en,ja-en,kk-en,ko-en,ky-en,lv-en,lt-en,mk-en,mg-en,ms-en,mr-en,ne-en,no-en,fa-en,pl-en,pt-en,ro-en,ru-en,sr-en,es-en,sv-en,th-en,tr-en,uk-en,ur-en,uz-en,vi-en --output_dir $NAME --model_id microsoft/Phi-3-medium-4k-instruct --save_steps 5000


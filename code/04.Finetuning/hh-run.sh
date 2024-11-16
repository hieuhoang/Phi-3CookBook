#!/usr/bin/env bash

export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface
#python hh-train-parallel.py --language_pairs sr-en --output_dir ./phi-3-medium-LoRA-sr-en --model_id microsoft/Phi-3-medium-4k-instruct
python hh-train-parallel.py --language_pairs is-en --output_dir ./phi-3-mini-LoRA-is-en --model_id microsoft/Phi-3-mini-4k-instruct
exit 1

#gpu2
# python hh-train-parallel.py --language_pairs af-en &
# python hh-train-parallel.py --language_pairs ar-en &
# python hh-train-parallel.py --language_pairs az-en &
# python hh-train-parallel.py --language_pairs bg-en &
# python hh-train-parallel.py --language_pairs ca-en &
# python hh-train-parallel.py --language_pairs zh-en &
# python hh-train-parallel.py --language_pairs cs-en &
# python hh-train-parallel.py --language_pairs da-en &
# python hh-train-parallel.py --language_pairs nl-en &
# python hh-train-parallel.py --language_pairs et-en &
# python hh-train-parallel.py --language_pairs fi-en &
# python hh-train-parallel.py --language_pairs fr-en &

#gpu1 ~/hh-run.2
# python hh-train-parallel.py --language_pairs gl-en &
# python hh-train-parallel.py --language_pairs ka-en &
# python hh-train-parallel.py --language_pairs de-en &
# python hh-train-parallel.py --language_pairs el-en &
# python hh-train-parallel.py --language_pairs gu-en &
# python hh-train-parallel.py --language_pairs he-en &
# python hh-train-parallel.py --language_pairs hi-en &
# python hh-train-parallel.py --language_pairs hu-en &
# python hh-train-parallel.py --language_pairs is-en &
# python hh-train-parallel.py --language_pairs id-en &
# python hh-train-parallel.py --language_pairs it-en &
# python hh-train-parallel.py --language_pairs ja-en &
# python hh-train-parallel.py --language_pairs kk-en &
# python hh-train-parallel.py --language_pairs ko-en &
# python hh-train-parallel.py --language_pairs ky-en &
# python hh-train-parallel.py --language_pairs lv-en &
# python hh-train-parallel.py --language_pairs lt-en &

#gpu2 ~/hh-run.3
# python hh-train-parallel.py --language_pairs mk-en &
# python hh-train-parallel.py --language_pairs mg-en &
# python hh-train-parallel.py --language_pairs ms-en &
# python hh-train-parallel.py --language_pairs mr-en &
# python hh-train-parallel.py --language_pairs ne-en &
# python hh-train-parallel.py --language_pairs no-en &
# python hh-train-parallel.py --language_pairs fa-en &
# python hh-train-parallel.py --language_pairs pl-en &
# python hh-train-parallel.py --language_pairs pt-en &
# python hh-train-parallel.py --language_pairs ro-en &
# python hh-train-parallel.py --language_pairs ru-en &
# python hh-train-parallel.py --language_pairs sr-en &
# python hh-train-parallel.py --language_pairs es-en &
# python hh-train-parallel.py --language_pairs sv-en &
# python hh-train-parallel.py --language_pairs th-en &
# python hh-train-parallel.py --language_pairs tr-en &
# python hh-train-parallel.py --language_pairs uk-en &
# python hh-train-parallel.py --language_pairs ur-en &
# python hh-train-parallel.py --language_pairs uz-en &
# python hh-train-parallel.py --language_pairs vi-en &
# wait

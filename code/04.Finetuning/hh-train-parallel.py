#!/usr/bin/env python
import os, sys
#os.environ['HF_HOME'] = '/mnt/eastus/hieu/workspace/cache/huggingface'
#export HF_HOME=/mnt/eastus/hieu/workspace/cache/huggingface

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
from random import randrange

# 'torch' is the PyTorch library, a popular open-source machine learning library for Python.
import torch

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library. 
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer

from huggingface_hub import notebook_login

# 'AutoPeftModelForCausalLM' is a class from the 'peft' library that provides a causal language model with PEFT (Performance Efficient Fine-Tuning) support.

from peft import AutoPeftModelForCausalLM

from datasets import concatenate_datasets, interleave_datasets, DatasetDict
import random
import argparse


##########################################################################################################
NLLB2Str = {
     "eng_Latn": "English",
     # Group 1:
    "dan_Latn": "Danish",
    "nld_Latn": "Dutch",
    "deu_Latn": "German",
    "isl_Latn": "Icelandic",
    "nob_Latn": "Norwegian",
    "swe_Latn": "Swedish",
    "afr_Latn": "Afrikaan",
    # Group 2:
    "cat_Latn": "Catalan",
    "ron_Latn": "Romanian",
    "glg_Latn": "Galician",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "spa_Latn": "Spanish",
    # Group 3:
    "bul_Cyrl": "Bulgarian",
    "mkd_Cyrl": "Macedonian",
    "srp_Cyrl": "Serbian",
    "ukr_Cyrl": "Ukrainian",
    "rus_Cyrl": "Russian",
    # Group 4:
    "ind_Latn": "Indonesian",
    "zsm_Latn": "Standard Malay",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "plt_Latn": "Plateau Malagasy",
    "fra_Latn": "French",
    # Group 5:
    "hun_Latn": "Hungarian",
    "ell_Grek": "Greek",
    "ces_Latn": "Czech",
    "pol_Latn": "Polish",
    "lit_Latn": "Lithuanian",
    "lvs_Latn": "Standard Latvian",
    # Group 6:
    "kat_Geor": "Georgian",
    "zho_Hans": "Chinese (Simplified)",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "fin_Latn": "Finnish",
    "est_Latn": "Estonian",
    # Group 7:
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "mar_Deva": "Marathi",
    "npi_Deva": "Nepali",
    "urd_Arab": "Urdu",
    # Group 8:
    "azj_Latn": "North Azerbaijani",
    "kaz_Cyrl": "Kazakh",
    "kir_Cyrl": "Kyrgyz",
    "tur_Latn": "Turkish",
    "uzn_Latn": "Northern Uzbek",
    "arb_Arab": "Modern Standard Arabic",
    "heb_Hebr": "Hebrew",
    "pes_Arab": "Western Persian",

}

NLLB_CODE = {
    "en": "eng_Latn",
    # Group 1:
    "da": "dan_Latn",
    "nl": "nld_Latn",
    "de": "deu_Latn",
    "is": "isl_Latn",
    "no": "nob_Latn",
    "sv": "swe_Latn",
    "af": "afr_Latn",
    # Group 2:
    "ca": "cat_Latn",
    "ro": "ron_Latn",
    "gl": "glg_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "es": "spa_Latn",
    # Group 3:
    "bg": "bul_Cyrl",
    "mk": "mkd_Cyrl",
    "sr": "srp_Cyrl",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    # Group 4:
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "mg": "plt_Latn",
    "fr": "fra_Latn",
    # Group 5:
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "cs": "ces_Latn",
    "pl": "pol_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    # Group 6:
    "ka": "kat_Geor",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fi": "fin_Latn",
    "et": "est_Latn",
    # Group 7:
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "ur": "urd_Arab",
    # Group 8:
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
    "ky": "kir_Cyrl",
    "tr": "tur_Latn",
    "uz": "uzn_Latn",
    "ar": "arb_Arab",
    "he": "heb_Hebr",
    "fa": "pes_Arab",
}

##########################################################################################################
# This code block defines two functions that are used to format the dataset for training a chat model.

# 'create_message_column' is a function that takes a row from the dataset and returns a dictionary 
# with a 'messages' key and a list of 'user' and 'assistant' messages as its value.
def create_message_column(row):
    # Initialize an empty list to store the messages.
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": f"{row['instruction']}\n Input: {row['input']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

def normalize_example(example):
    lg1, lg2 = example["translation"].keys()
    if random.random() < 0.5:
        combined_translation = example["translation"][lg1] + " " + example["translation"][lg2]
    else:
        combined_translation = example["translation"][lg2] + " " + example["translation"][lg1]
    return {
        "raw_text": combined_translation,
    }

##########################################################################################################
# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary 
# with a 'text' key and a string of formatted chat messages as its value.
def format_dataset_chatml(example, tokenizer):
    nllb1, nllb2 = example["translation"].keys()

    if random.random() < 0.5:
        lg1, lg2 = NLLB2Str[nllb1], NLLB2Str[nllb2]
        sent1 = example["translation"][nllb1]
        sent2 = example["translation"][nllb2]
    else:
        lg2, lg1 = NLLB2Str[nllb1], NLLB2Str[nllb2]
        sent2 = example["translation"][nllb1]
        sent1 = example["translation"][nllb2]
    
    userContent = f"Translate the following sentence from {lg1} to {lg2}:\n{sent1}"
    messages = [{"content": userContent, "role": "user"}, 
                {"content": sent2, "role": "assistant"}]

    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    ret = {"text": tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)}
    return ret

def format_dataset_chatml_th(example, tokenizer):
    iso1, iso2 = example["translation"].keys()

    if random.random() < 0.5:
        lg1, lg2 = NLLB2Str[NLLB_CODE[iso1]], NLLB2Str[NLLB_CODE[iso2]]
        sent1 = example["translation"][iso1]
        sent2 = example["translation"][iso2]
    else:
        lg2, lg1 = NLLB2Str[NLLB_CODE[iso1]], NLLB2Str[NLLB_CODE[iso2]]
        sent2 = example["translation"][iso1]
        sent1 = example["translation"][iso2]
    
    userContent = f"Translate the following sentence from {lg1} to {lg2}:\n{sent1}"
    messages = [{"content": userContent, "role": "user"}, 
                {"content": sent2, "role": "assistant"}]

    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    ret = {"text": tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)}
    return ret

def TrainValidTest(chatTrainOrig, validSize, testSize):
    assert chatTrainOrig.num_rows > (validSize + testSize)

    r1 = (validSize + testSize) / chatTrainOrig.num_rows
    t1 = chatTrainOrig.train_test_split(test_size=r1)

    r2 = testSize / (validSize + testSize)
    t2 = t1["test"].train_test_split(test_size=r2)
    t1["test"] = t2["test"]
    t1["validation"] = t2["train"]
    return t1

def LanguagePairs2NLLBList(str):
    ret = []
    lps = str.split(",")

    for lp in lps:
        toks = lp.split("-")
        assert len(toks) == 2
        src_lang, tgt_lang = NLLB_CODE[toks[0]], NLLB_CODE[toks[1]]
        language_key = f"{src_lang}-{tgt_lang}" if src_lang < tgt_lang else f"{tgt_lang}-{src_lang}"
        ret.append(language_key)

    return ret

##########################################################################################################

def main():
    argParse = argparse.ArgumentParser(description="blah")
    argParse.add_argument("--output_dir", required=True)
    argParse.add_argument("--model_id", required=True)
    argParse.add_argument("--language_pairs", required=True)
    argParse.add_argument("--nllb_interleave_probs", default=None)
    argParse.add_argument("--nllb_interleave_probs", default=None)
    argParse.add_argument("--max_steps", type=int, default=999999)



    args = argParse.parse_args()

    languagePairs = LanguagePairs2NLLBList(args.language_pairs)
    # 'model_id' and 'model_name' are the identifiers for the pre-trained model that you want to fine-tune. 
    # In this case, it's the 'Phi-3-mini-4k-instruct' model from Microsoft.
    # Model Names 
    # microsoft/Phi-3-mini-4k-instruct
    # microsoft/Phi-3-mini-128k-instruct
    # microsoft/Phi-3-small-8k-instruct
    # microsoft/Phi-3-small-128k-instruct
    # microsoft/Phi-3-medium-4k-instruct
    # microsoft/Phi-3-medium-128k-instruct
    # microsoft/Phi-3-vision-128k-instruct
    # microsoft/Phi-3-mini-4k-instruct-onnx
    # microsoft/Phi-3-mini-4k-instruct-onnx-web
    # microsoft/Phi-3-mini-128k-instruct-onnx
    # microsoft/Phi-3-small-8k-instruct-onnx-cuda
    # microsoft/Phi-3-small-128k-instruct-onnx-cuda
    # microsoft/Phi-3-medium-4k-instruct-onnx-cpu
    # microsoft/Phi-3-medium-4k-instruct-onnx-cuda
    # microsoft/Phi-3-medium-4k-instruct-onnx-directml
    # microsoft/Phi-3-medium-128k-instruct-onnx-cpu
    # microsoft/Phi-3-medium-128k-instruct-onnx-cuda
    # microsoft/Phi-3-medium-128k-instruct-onnx-directml
    # microsoft/Phi-3-mini-4k-instruct-gguf

    model_name = args.model_id 

    # 'new_model' is the name that you want to give to the fine-tuned model.
    new_model = "Name of your new model"

    # 'hf_model_repo' is the repository on the Hugging Face Model Hub where the fine-tuned model will be saved. Update UserName to your Hugging Face Username
    hf_model_repo="UserName/"+new_model

    # 'device_map' is a dictionary that maps the model to the GPU device. 
    # In this case, the entire model is loaded on GPU 0.
    device_map = "auto" #{"": 0}

    # The following are parameters for the LoRA (Learning from Random Architecture) model.

    # 'lora_r' is the dimension of the LoRA attention.
    lora_r = 16

    # 'lora_alpha' is the alpha parameter for LoRA scaling.
    lora_alpha = 16

    # 'lora_dropout' is the dropout probability for LoRA layers.
    lora_dropout = 0.05

    # 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]

    # 'set_seed' is a function that sets the seed for generating random numbers, 
    # which is used for reproducibility of the results.
    seed = 1001
    set_seed(seed)


    # The 'len' function is used to get the size of the dataset, which is then printed.
    #print(f"dataset size: {len(dataset)}")

    # 'randrange' is a function from the 'random' module that generates a random number within the specified range.
    # Here it's used to select a random example from the dataset, which is then printed.
    #print(dataset[randrange(len(dataset))])

    # This code block is used to load a pre-trained model and its associated tokenizer from the Hugging Face Model Hub.

    # 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
    # 'model_id' is passed as an argument to specify which tokenizer to load.
    # 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
    # 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
    # 'use_fast' is set to True to use the fast version of the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)

    # The padding token is set to the unknown token.
    tokenizer.pad_token = tokenizer.unk_token

    # The ID of the padding token is set to the ID of the unknown token.
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
    tokenizer.padding_side = 'left'

    # NNLB
    train_raw_data = {}
    nllb_raw_data = []

    if args.nllb_interleave_probs:
        interleave_probs = [float(p) for p in args.nllb_interleave_probs.split(",")]
    else:
        interleave_probs = [1/len(languagePairs)] * len(languagePairs)

    nllb_pretrain_data_path = "allenai/nllb"
    load_kwargs = {'cache_dir': None, 'use_auth_token': None, 'streaming': False, 'trust_remote_code': True}

    for language_key in languagePairs:
        set_seed(seed)
        if language_key == 'eng_Latn-tha_Thai':
            dataDictOrig = load_dataset(
                "Helsinki-NLP/opus-100",
                "en-th",
                **load_kwargs,
            )
            chatDict = DatasetDict()
            train = dataDictOrig["train"].map(lambda batch: format_dataset_chatml_th(batch, tokenizer), remove_columns=dataDictOrig["train"].column_names)
            valid = dataDictOrig["validation"].map(lambda batch: format_dataset_chatml_th(batch, tokenizer), remove_columns=dataDictOrig["validation"].column_names)
            test = dataDictOrig["test"].map(lambda batch: format_dataset_chatml_th(batch, tokenizer), remove_columns=dataDictOrig["test"].column_names)
            
            chatDict["train"] = train
            chatDict["validation"] = valid
            chatDict["test"] = test
        else:
            dataDictOrig = load_dataset(nllb_pretrain_data_path, language_key, **load_kwargs)
                
            trainOrig = dataDictOrig["train"]
            chatTrainOrig = trainOrig.map(lambda batch: format_dataset_chatml(batch, tokenizer), remove_columns=trainOrig.column_names)

            set_seed(seed)
            chatDict = TrainValidTest(chatTrainOrig, 5000, 5000)
            
        chatTrain = chatDict["train"]

        set_seed(seed)
        chatTrain = chatTrain.shuffle(seed=seed)
        #print("lg_dataset1", lg_dataset, type(lg_dataset)) #, lg_dataset['translation'])


        nllb_raw_data.append(chatTrain)

    set_seed(seed)
    train_raw_data["nllb_pretrain"] = interleave_datasets(nllb_raw_data, probabilities=interleave_probs, seed=seed, stopping_strategy="first_exhausted")
    #train_raw_data["nllb_pretrain"] = nllb_raw_data[0]
    print("train_raw_data", train_raw_data["nllb_pretrain"], type(train_raw_data["nllb_pretrain"]))
    #exit(0)

    # This code block is used to set the compute data type and attention implementation based on whether bfloat16 is supported on the current CUDA device.

    # 'torch.cuda.is_bf16_supported()' is a function that checks if bfloat16 is supported on the current CUDA device.
    # If bfloat16 is supported, 'compute_dtype' is set to 'torch.bfloat16' and 'attn_implementation' is set to 'flash_attention_2'.
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        #attn_implementation = 'flash_attention_2'
        attn_implementation = 'eager'

        # If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

    # This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
    print(attn_implementation)

    # 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
    # 'model_id' is passed as an argument to specify which model to load.
    # 'torch_dtype' is set to the compute data type determined earlier.
    # 'trust_remote_code' is set to True to trust the remote code in the model files.
    # 'device_map' is passed as an argument to specify the device mapping for distributed training.
    # 'attn_implementation' is set to the attention implementation determined earlier.
    model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=compute_dtype, trust_remote_code=True, device_map=device_map,
            attn_implementation=attn_implementation
    )


    # This code block is used to define the training arguments for the model.

    # 'TrainingArguments' is a class that holds the arguments for training a model.
    # 'output_dir' is the directory where the model and its checkpoints will be saved.
    # 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
    # 'do_eval' is set to True, meaning that evaluation will be performed.
    # 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
    # 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
    # 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
    # 'log_level' is set to "debug", meaning that all log messages will be printed.
    # 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
    # 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
    # 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
    # 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
    # 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
    # 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
    # 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
    # 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
    # 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
    # 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
    # 'seed' is set to 42, which is the seed for the random number generator.

    # LoraConfig object is created with the following parameters:
    # 'r' (rank of the low-rank approximation) is set to 16,
    # 'lora_alpha' (scaling factor) is set to 16,
    # 'lora_dropout' dropout probability for Lora layers is set to 0.05,
    # 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
    # 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


    args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=16,
            log_level="debug",
            save_strategy="steps",
            logging_steps=500,
            learning_rate=1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=100,
            num_train_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            report_to="none",
            seed=seed,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
    )

    peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
    )


    # This code block is used to initialize the SFTTrainer, which is used to train the model.

    # 'model' is the model that will be trained.
    # 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
    # 'peft_config' is the configuration for peft, which is used for instruction tuning.
    # 'dataset_text_field' is set to "text", meaning that the 'text' field of the dataset will be used as the input for the model.
    # 'max_seq_length' is set to 512, meaning that the maximum length of the sequences that will be fed to the model is 512 tokens.
    # 'tokenizer' is the tokenizer that will be used to tokenize the input text.
    # 'args' are the training arguments that were defined earlier.

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_raw_data["nllb_pretrain"],
            eval_dataset=chatDict["validation"],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=args,
    )

    # This code block is used to train the model and save it locally.

    # 'trainer.train()' is a method that starts the training of the model.
    # It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.
    trainer.train()

    # 'trainer.save_model()' is a method that saves the trained model locally.
    # The model will be saved in the directory specified by 'output_dir' in the training arguments.
    trainer.save_model()

    # This code block is used to free up GPU memory.

    # 'del model' and 'del trainer' are used to delete the 'model' and 'trainer' objects. 
    # This removes the references to these objects, allowing Python's garbage collector to free up the memory they were using.

    del model
    del trainer

    # 'import gc' is used to import Python's garbage collector module.
    import gc

    # 'gc.collect()' is a method that triggers a full garbage collection, which can help to free up memory.
    # It's called twice here to ensure that all unreachable objects are collected.
    gc.collect()
    gc.collect()

    # 'torch.cuda.empty_cache()' is a PyTorch method that releases all unoccupied cached memory currently held by 
    # the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
    torch.cuda.empty_cache()

    # 'gc.collect()' is a method that triggers a full garbage collection in Python.
    # It forces the garbage collector to release unreferenced memory, which can be helpful in managing memory usage, especially in a resource-constrained environment.
    gc.collect()

    # 'AutoPeftModelForCausalLM.from_pretrained' is a method that loads a pre-trained model (adapter model) and its base model.
    #  The adapter model is loaded from 'args.output_dir', which is the directory where the trained model was saved.
    # 'low_cpu_mem_usage' is set to True, which means that the model will use less CPU memory.
    # 'return_dict' is set to True, which means that the model will return a 'ModelOutput' (a named tuple) instead of a plain tuple.
    # 'torch_dtype' is set to 'torch.bfloat16', which means that the model will use bfloat16 precision for its computations.
    # 'trust_remote_code' is set to True, which means that the model will trust and execute remote code.
    # 'device_map' is the device map that will be used by the model.

    new_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16, #torch.float16,
        trust_remote_code=True,
        device_map=device_map,
    )

    # 'new_model.merge_and_unload' is a method that merges the model and unloads it from memory.
    # The merged model is stored in 'merged_model'.

    merged_model = new_model.merge_and_unload()

    # 'merged_model.save_pretrained' is a method that saves the merged model.
    # The model is saved in the directory "merged_model".
    # 'trust_remote_code' is set to True, which means that the model will trust and execute remote code.
    # 'safe_serialization' is set to True, which means that the model will use safe serialization.

    merged_model.save_pretrained(f"{args.output_dir}/merged_model", trust_remote_code=True, safe_serialization=True)

    # 'tokenizer.save_pretrained' is a method that saves the tokenizer.
    # The tokenizer is saved in the directory "merged_model".

    tokenizer.save_pretrained(f"{args.output_dir}/merged_model")

if __name__ == '__main__':
    sys.exit(main())

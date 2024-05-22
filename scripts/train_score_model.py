from datasets import load_dataset

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from model import load_model, load_tokenizer
from fast_detect_gpt import evaluation_sft_model
from dna_gpt import evaluation_sft_model_dna
from detect_gpt import evaluation_sft_model_detect
import torch
import datasets

def filter_by_english_and_llama3(sample):
    return len(sample["conversation"][0]["content"]) < 480


def filter_by_english_and_version_3_5(sample):
    return sample["language"] == 'English' \
            and sample["model"] == "gpt-3.5-turbo" and len(sample["conversation"][0]["content"]) < 480

def filter_by_english_and_version(sample):
    # return sample["language"] == 'German' and sample["model"] == "gpt-4" and len(sample["conversation"][0]["content"]) < 512
    return sample["language"] == 'English' and 6 <= sample["timestamp"].month <= 11 \
            and sample["model"] == "gpt-4" and len(sample["conversation"][0]["content"]) < 512
    # return sample["language"] == 'English' \
    #         and (sample["model"] == "gpt-4" or sample["model"] == "gpt-3.5-turbo") and len(sample["conversation"][0]["content"]) < 512
def filter_claude(sample):
    return len(sample["prompt"]) < 2048


class evaluation_callback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # For FastDetctGPT
        # Test on Llama3-8b
        evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_llama3/data/xsum_llama3-8b')
        evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_llama3/data/writing_llama3-8b')
        evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_llama3/data/pubmed_llama3-8b')

        # Test on Claude
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/xsum_claude-3-opus-20240229')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/writing_claude-3-opus-20240229')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/pubmed_claude-3-opus-20240229')
        
        # Test on GPT-4
        # On 0613
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/xsum_gpt-4-0613')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/writing_gpt-4-0613')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/pubmed_gpt-4-0613')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='german', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/german_gpt-4-0613')
        # On 1106
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-1106preview/data/xsum_gpt-4-1106-preview')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-1106preview/data/writing_gpt-4-1106-preview')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-1106preview/data/pubmed_gpt-4-1106-preview')
        # On 0105
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-20240125preview/data/xsum_gpt-4-0125-preview')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-20240125preview/data/writing_gpt-4-0125-preview')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-20240125preview/data/pubmed_gpt-4-0125-preview')
        # On 0409
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-turbo20240409preview/data/xsum_gpt-4-turbo-2024-04-09')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-turbo20240409preview/data/writing_gpt-4-turbo-2024-04-09')
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-turbo20240409preview/data/pubmed_gpt-4-turbo-2024-04-09')
        
        # # For GPT-3.5, we only do testing on pubmed
        # On 0301
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0301/data/pubmed_gpt-3.5-turbo-0301')
        # On 0613
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0613/data/pubmed_gpt-3.5-turbo-0613')
        # On 0613
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo1106/data/pubmed_gpt-3.5-turbo-1106')
        # On 0613
        # evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo20240125/data/pubmed_gpt-3.5-turbo-0125')

        #For DNA-GPT
        # Test on Claude
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/xsum_claude-3-opus-20240229')
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/writing_claude-3-opus-20240229')
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/pubmed_claude-3-opus-20240229')
        # Test on GPT-4
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/xsum_gpt-4-0613')
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/writing_gpt-4-0613')
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/pubmed_gpt-4-0613')
        
        # # For GPT-3.5, we only do testing on pubmed
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0301/data/pubmed_gpt-3.5-turbo-0301')
        
        #For Detect-GPT
        # Test on Claude
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/xsum_claude-3-opus-20240229')
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/writing_claude-3-opus-20240229')
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_claude3opus-20240229/data/pubmed_claude-3-opus-20240229')
        # # Test on GPT-4
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='xsum', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/xsum_gpt-4-0613')
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='writing', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/writing_gpt-4-0613')
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/pubmed_gpt-4-0613')
        
        # # # For GPT-3.5, we only do testing on pubmed
        # evaluation_sft_model_detect(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt3.5turbo0301/data/pubmed_gpt-3.5-turbo-0301')
        
        # For Adver Attack
        # rank = torch.distributed.get_rank() 
        # if rank == 0:
        #     # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/pubmed_gpt-4-0613.raw_data.json')
        #     evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1.raw_data_0.1.json')
        # else:
        #     evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1._percent_0.5.raw_data.json')

        # elif rank == 1:
        #     # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1._percent_0.2.raw_data.json')
        # elif rank == 2:
        #     # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1._percent_0.3.raw_data.json')
        # elif rank == 3:
        #     # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1._percent_0.4.raw_data.json')
        # evaluation_sft_model_dna(kwargs["model"], kwargs["tokenizer"], state.epoch, dataset='pubmed', dataset_file='/home/dongk/dkgroup/congzeng/fast-detect-gpt/exp_gpt4-0613/data/replaced_pubmed_gpt-4-0613.t5-3b.perturbation_1._percent_0.5.raw_data.json')

        return 


def build_sft_dataset(name, num_sample, filter_fn=None):
    if name == "wildchat":
        dataset = load_dataset("allenai/WildChat", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "claude":
        dataset = load_dataset("Sao10K/Claude-3-Opus-Instruct-5K", 'Instruct Data v1 - Merged', split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    elif name == "llama3":
        dataset = load_dataset("Shengkun/llama3_texts", split="train")
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn)
    return dataset.train_test_split(train_size=num_sample)["train"]


if __name__ == "__main__":
    # load data
    
    # dataset = load_dataset("allenai/WildChat", split="train")
    # # dataset = dataset.train_test_split(train_size=10000)["train"]
    # dataset = dataset.filter(filter_by_english_and_version)
    # dataset = dataset.train_test_split(train_size=5000)["train"]
    # dataset = dataset.train_test_split(train_size=100)

    
    name = "wildchat"
    # dataset_4 = build_sft_dataset(name=name, num_sample=5000, filter_fn=filter_by_english_and_version)
    # dataset_3_5 = build_sft_dataset(name=name, num_sample=5000, filter_fn=filter_by_english_and_version_3_5)
    # dataset_claude = build_sft_dataset(name="claude", num_sample=3000, filter_fn=filter_claude)
    dataset_llama3 = build_sft_dataset(name="llama3", num_sample=4000, filter_fn=filter_by_english_and_llama3)

    


    tokenizer = load_tokenizer("llama2-7b", for_dataset="WildChat", cache_dir="./ckpt") # Check the padding method of WildChat
    #generate prompt
    cutoff_len = 1024
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_instruct_token(prompt, prompt_response_comb):
        prompts_ids = tokenizer(
            text=prompt,
            return_tensors=None,
        )

        combined_target_ids = tokenizer(
            text=prompt_response_comb,
            return_tensors=None,
            padding="max_length", max_length=cutoff_len, truncation=True
        )                    

        prompt_len = len(prompts_ids["input_ids"])
        labels = [-100] * prompt_len + combined_target_ids["input_ids"][prompt_len:]
        combined_target_ids["labels"] = labels

        return combined_target_ids

    def generate_gpt_input_and_tokenize(sample):
        conversation = sample["conversation"]
        prompt = conversation[0]['content']
        prompt_response = f"{conversation[0]['content']}{conversation[1]['content']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)
        
    def generate_claude_input_and_tokenize(sample):
        prompt = sample["prompt"]
        prompt_response = f"{sample['prompt']}{sample['response']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)

    def generate_llama3_input_and_tokenize(sample):
        prompt = sample["llama3_8b_prompt"]
        prompt_response = f"{sample['llama3_8b_prompt']}{sample['llama3_8b_output']}"
        return generate_instruct_token(prompt=prompt, prompt_response_comb=prompt_response)

    # token_dataset_4 = dataset_4.map(generate_gpt_input_and_tokenize)
    # tokenized_dataset = dataset_3_5.map(generate_gpt_input_and_tokenize)
    token_dataset_llama3 = dataset_llama3.map(generate_llama3_input_and_tokenize)
    # token_dataset_claude = dataset_claude.map(generate_claude_input_and_tokenize)
    # if name == "wildchat":
    #     tokenized_dataset = dataset.map(generate_gpt_input_and_tokenize)
    # elif name == "claude":
    #     tokenized_dataset = dataset.map(generate_claude_input_and_tokenize)

    tokenized_dataset = datasets.concatenate_datasets([token_dataset_llama3])
    tokenized_dataset = tokenized_dataset.remove_columns(dataset_llama3.column_names)
    # tokenized_dataset = tokenized_dataset.remove_columns(dataset_claude.column_names)

    # tokenized_dataset = tokenized_dataset.remove_columns(["prompt", "response", "tokens", "id"])

    # load tokenizer and model e.g. llama2-7b, gpt-neo-2.7B
    model = load_model("llama2-7b", device="cpu", cache_dir="./ckpt")

    # for llama-2-7B
    # target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    # for Gpt-neo
    # target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "c_fc", " c_proj"],
    
    # load LoRA model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        fan_in_fan_out=False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    #Train
    args = TrainingArguments(
        output_dir="./ckpt",
        remove_unused_columns=False,
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        bf16=True,
        num_train_epochs=1,
        logging_steps=1,
        do_eval=False,
        )
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=False),
        callbacks=[evaluation_callback])
    torch.cuda.empty_cache()
    trainer.train()

    #Test every epoch





    
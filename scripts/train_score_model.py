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


def filter_by_english_and_version(sample):
    return sample["language"] == 'English' and 3 <= sample["timestamp"].month <= 6 \
            and 1 <= sample["timestamp"].day < 13 and sample["turn"] == 1 \
            and sample["model"] == "gpt-3.5-turbo"

class evaluation_callback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        evaluation_sft_model(kwargs["model"], kwargs["tokenizer"], TrainerState.epoch)
        return 


if __name__ == "__main__":
    # load data
    dataset = load_dataset("allenai/WildChat", split="train")
    dataset = dataset.filter(filter_by_english_and_version)
    dataset = dataset.train_test_split(train_size=5000)["train"]
    # dataset = dataset.train_test_split(train_size=100)

    tokenizer = load_tokenizer("llama2-7b", for_dataset="WildChat", cache_dir="./ckpt") # Check the padding method of WildChat
    #generate prompt
    cutoff_len = 512
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
    
    def generate_input_and_tokenize(sample):
        conversation = sample["conversation"]
        prompt = f"{conversation[0]['content']}{conversation[1]['content']}"
        prompt_tokenized = tokenize(prompt)
        return prompt_tokenized

    tokenized_dataset = dataset.map(generate_input_and_tokenize)
    tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
    # tokenized_dataset = tokenized_dataset.remove_columns(["token_type_ids"])

    #load tokenizer and model e.g. llama2-7B
    model = load_model("llama2-7b", device="cpu", cache_dir="./ckpt")

    #load LoRA model
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
        num_train_epochs=10,
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
    trainer.train()

    #Test every epoch





    
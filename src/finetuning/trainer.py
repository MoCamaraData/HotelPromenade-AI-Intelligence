# finetuning/trainer.py
from unsloth import FastLanguageModel
from typing import Dict, Any
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer



# ============================================================
# STEP 1 — Load Base Model
# ============================================================

def load_base_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype=None,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    return model, tokenizer


# ============================================================
# STEP 2 — Format Dataset with Chat Template
# ============================================================

def format_dataset_with_template(
    jsonl_path: str,
    tokenizer,
    add_generation_prompt: bool = False,
):
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def apply_template(example: Dict[str, Any]):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return {"text": text}

    dataset = dataset.map(
        apply_template,
        remove_columns=dataset.column_names,
        num_proc=1,
        load_from_cache_file=False,
        desc="Applying chat template to dataset",
    )

    return dataset


# ============================================================
# STEP 3 — Apply LoRA
# ============================================================

def apply_lora(
    model,
    r: int,
    lora_alpha: int,
    target_modules,
    lora_dropout: float,
    bias: str,
    use_gradient_checkpointing: str,
    random_state: int,
):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )

    model.print_trainable_parameters()
    return model


# ============================================================
# STEP 4 — Create Trainer
# ============================================================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    num_train_epochs: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    logging_steps: int,
    save_strategy: str,
    seed: int,
    max_seq_length: int,
    packing: bool,
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=packing,
    )

    return trainer


# ============================================================
# STEP 5 — Train
# ============================================================

def train_model(trainer):
    result = trainer.train()
    print(f"Final training loss: {result.training_loss:.4f}")
    return result


# ============================================================
# STEP 6 — Save
# ============================================================

def save_model(trainer, tokenizer, output_dir: str):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
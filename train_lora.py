import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/lora_train.jsonl")

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)

    # LoRA config: start small
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # good minimal set
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files={"train": TRAIN_FILE})
    split = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    def apply_chat_template(messages, add_generation_prompt=False):
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Fallback to a minimal template compatible with TinyLlama-style tags.
        text = ""
        for msg in messages:
            if msg["role"] == "system":
                text += f"<|system|>\n{msg['content']}\n"
            elif msg["role"] == "user":
                text += f"<|user|>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                text += f"<|assistant|>\n{msg['content']}\n"
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return text

    def format_and_tokenize(ex):
        messages = [{"role": "user", "content": ex["prompt"]}]
        prompt_text = apply_chat_template(messages, add_generation_prompt=True)
        full_text = apply_chat_template(
            messages + [{"role": "assistant", "content": ex["response"]}],
            add_generation_prompt=False,
        )

        prompt_ids = tok(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full = tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        )

        labels = full["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        full["labels"] = labels
        return full

    tokenized_train = train_ds.map(
        format_and_tokenize, remove_columns=train_ds.column_names
    )
    tokenized_eval = eval_ds.map(
        format_and_tokenize, remove_columns=eval_ds.column_names
    )

    args = TrainingArguments(
        output_dir="out_lora",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=10,     # tiny dataset => more epochs is ok
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, label_pad_token_id=-100
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    trainer.train()

    model.save_pretrained("out_lora")
    tok.save_pretrained("out_lora")
    print("Saved LoRA adapter to out_lora/")

if __name__ == "__main__":
    main()

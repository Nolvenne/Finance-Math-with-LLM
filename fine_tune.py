from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load model & tokenizer
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="./offload",
    torch_dtype=torch.float16
)

dataset = load_dataset("json", data_files={"train": "fine_data.jsonl"})

# Tokenization
def tokenize_fn(batch):
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"]
)

# Train the model
trainer.train()

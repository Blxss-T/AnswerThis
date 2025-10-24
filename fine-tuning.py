from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "deepset/xlm-roberta-large-squad2"  # same as in model.py
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("json", data_files="maternal_dataset_squad.json")

# Tokenize
def preprocess(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized = dataset.map(preprocess, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
)

trainer.train()

model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print(" Fine-tuning complete! Saved to ./fine_tuned_maternal_model")

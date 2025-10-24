from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load model and tokenizer
model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load dataset (your JSON has top-level "data")
raw = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")

# Extract the real data
dataset = raw["train"]["data"]

# --- Flatten into question–context–answer ---
contexts, questions, start_positions, end_positions = [], [], [], []
for article in dataset:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]
            start = answer["answer_start"]
            end = start + len(answer["text"])
            contexts.append(context)
            questions.append(question)
            start_positions.append(start)
            end_positions.append(end)

# --- Convert to Dataset ---
from datasets import Dataset
flat_dataset = Dataset.from_dict({
    "context": contexts,
    "question": questions,
    "start_positions": start_positions,
    "end_positions": end_positions
})

# --- Tokenization ---
def preprocess_function(examples):
    encodings = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )
    encodings["start_positions"] = examples["start_positions"]
    encodings["end_positions"] = examples["end_positions"]
    return encodings

tokenized = flat_dataset.map(preprocess_function, batched=True)

# --- Split into train/test ---
dataset_split = tokenized.train_test_split(test_size=0.1)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print("Fine-tuning complete! Model saved to ./fine_tuned_maternal_model")

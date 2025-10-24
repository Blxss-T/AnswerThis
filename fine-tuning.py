from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load model and tokenizer
model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load dataset
raw_dataset = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")

# --- Flatten the nested SQuAD structure ---
def flatten_squad(example):
    contexts, questions, start_positions, end_positions = [], [], [], []
    for article in example["data"]:
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
    return {"context": contexts, "question": questions, "start_positions": start_positions, "end_positions": end_positions}

# Apply flattening
flat_dataset = raw_dataset["train"].map(flatten_squad, batched=False, remove_columns=raw_dataset["train"].column_names)

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

tokenized_data = flat_dataset.map(preprocess_function, batched=True)

# --- Split train/validation ---
dataset_dict = tokenized_data.train_test_split(test_size=0.1)

# --- Training setup ---
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
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
)

# --- Start training ---
trainer.train()

# --- Save fine-tuned model ---
model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print("Fine-tuning complete! Model saved to ./fine_tuned_maternal_model")

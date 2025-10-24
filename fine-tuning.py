from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load the dataset (SQuAD format)
dataset = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")

# Function to preprocess each example
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    start_positions = [ans["answer_start"][0] for ans in answers]
    end_positions = [
        ans["answer_start"][0] + len(ans["text"][0]) for ans in answers
    ]

    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=384,
    )

    encodings["start_positions"] = start_positions
    encodings["end_positions"] = end_positions
    return encodings

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training setup
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
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print("Fine-tuning complete! Saved to ./fine_tuned_maternal_model")

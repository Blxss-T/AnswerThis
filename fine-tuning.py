from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load model and tokenizer
model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load dataset (SQuAD style)
raw = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")
data = raw["train"]

# --- Handle nested structure safely ---
if "data" in data.column_names:
    dataset = data[0]["data"]  # unwrap top-level
else:
    dataset = data  # already at paragraph level

# --- Flatten structure ---
contexts, questions, start_positions, end_positions = [], [], [], []

for article in dataset:
    if isinstance(article, dict) and "paragraphs" in article:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answer = qa["answers"][0]
                questions.append(qa["question"])
                contexts.append(context)
                start_positions.append(answer["answer_start"])
                end_positions.append(answer["answer_start"] + len(answer["text"]))
    elif isinstance(article, list):  # if loaded as list
        for sub_article in article:
            for paragraph in sub_article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]
                    questions.append(qa["question"])
                    contexts.append(context)
                    start_positions.append(answer["answer_start"])
                    end_positions.append(answer["answer_start"] + len(answer["text"]))

# --- Convert to dataset ---
flat_dataset = Dataset.from_dict({
    "context": contexts,
    "question": questions,
    "start_positions": start_positions,
    "end_positions": end_positions
})

# --- Tokenize ---
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

# --- Split ---
dataset_split = tokenized.train_test_split(test_size=0.1)

# --- Train ---
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

# --- Save ---
model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print(" Fine-tuning complete! Model saved to ./fine_tuned_maternal_model")

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate

# --- Load base model and tokenizer ---
model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# --- Load maternal dataset (SQuAD format) ---
raw = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")
data = raw["train"]

# --- Unwrap nested structure if needed ---
if "data" in data.column_names:
    dataset = data[0]["data"]
else:
    dataset = data

# --- Flatten into simple lists ---
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
    elif isinstance(article, list):
        for sub_article in article:
            for paragraph in sub_article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]
                    questions.append(qa["question"])
                    contexts.append(context)
                    start_positions.append(answer["answer_start"])
                    end_positions.append(answer["answer_start"] + len(answer["text"]))

flat_dataset = Dataset.from_dict({
    "context": contexts,
    "question": questions,
    "start_positions": start_positions,
    "end_positions": end_positions
})

# --- Preprocessing with correct token alignment ---
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    encodings = tokenizer(
        questions,
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(encodings["offset_mapping"]):
        start_char = examples["start_positions"][i]
        end_char = examples["end_positions"][i]
        sequence_ids = encodings.sequence_ids(i)

        # Locate the context within tokens
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1)

        token_start = token_end = None
        for j in range(context_start, context_end):
            start, end = offsets[j]
            if start <= start_char < end:
                token_start = j
            if start < end_char <= end:
                token_end = j

        # Fallback in case truncation happens
        if token_start is None: token_start = context_start
        if token_end is None: token_end = context_end - 1

        start_positions.append(token_start)
        end_positions.append(token_end)

    encodings.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    encodings.pop("offset_mapping")
    return encodings

tokenized = flat_dataset.map(preprocess_function, batched=True)

# --- Split into train and test sets ---
dataset_split = tokenized.train_test_split(test_size=0.1)

# --- Evaluation metrics (F1 + Exact Match) ---
metric = evaluate.load("squad")

def compute_metrics(p):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

# --- Training setup ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    compute_metrics=None,  # You can replace None with compute_metrics if desired
)

# --- Train and Save ---
trainer.train()

model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print("✨ Fine-tuning complete! Model saved to './fine_tuned_maternal_model'")

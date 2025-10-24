from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")

# Flatten SQuAD structure
def preprocess_function(example):
    questions, contexts, start_positions, end_positions = [], [], [], []

    for article in example["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]
                questions.append(question)
                contexts.append(context)
                start_positions.append(answer["answer_start"])
                end_positions.append(answer["answer_start"] + len(answer["text"]))

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
tokenized_datasets = dataset.map(preprocess_function, batched=False)

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

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("json", data_files="maternal_finetuning-dataset-squad.json")["train"]

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

# Preprocess the dataset
tokenized_dataset = preprocess_function(dataset)

# Convert to Hugging Face DatasetDict
from datasets import Dataset
full_dataset = Dataset.from_dict(tokenized_dataset)

# Split into train and validation
dataset_dict = full_dataset.train_test_split(test_size=0.1)
print(dataset_dict)

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
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],  # validation dataset
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_maternal_model")
tokenizer.save_pretrained("./fine_tuned_maternal_model")

print("Fine-tuning complete! Saved to ./fine_tuned_maternal_model")

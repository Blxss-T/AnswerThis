from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json

# Loading the pre-trained model ("mT5 small")
model_name = "mT5 small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Q&A dataset
with open("maternity_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)
    
# Combine all answers into one long text
context = " ".join([item["answer"] for item in qa_data])

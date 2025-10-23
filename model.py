from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json

# Loading the pre-trained model ("mT5 small")
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Q&A dataset
with open("maternal_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Combine all answers into one long text
context = " ".join([item["answer"] for item in qa_data])

# Create a pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Run a loop to ask questions

while True:
    choice=input("1.Gukomeza kubaza: 00.Gusubira inyuma: 3.Gusohokamo")
if choice == "1":
    while True:
    question = input("Shyiramo ikibazo: ") 
    if question == "0":
        break
    result = qa_pipeline(question=question, context=context)
    print("Igisubizo:", result['answer'],"\n")
     elif choice == "2":
        print("Wahisemo gusubira inyuma...")
        # Aha ushobora gushyiraho indi menu cyangwa ibikorwa byasubirwamo.
        continue
     elif choice == "3":
        print("Murakoze! Muri gusohoka...")
        break
     else:
        print("Amahitamo ntakwiye, ongera ugerageze.\n")


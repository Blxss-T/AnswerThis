from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json

# Load pre-trained multilingual QA model
model_name = "xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Q&A dataset
with open("maternal_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Combine all answers into one long text
context = " ".join([item["answer"] for item in qa_data])

# Function to split long text into chunks (to improve QA accuracy)
def chunk_text(text, max_tokens=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

# Create QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
print("========= Murakaza neza kuri MotherLink Baza! =========")

# Main interaction loop
while True:
    choice = input("1. Gukomeza kubaza | 2. Gusubira inyuma | 3. Gusohokamo: ")

    if choice == "1":
        while True:
            question = input("Shyiramo ikibazo (andika '0' kugira usubire inyuma): ")
            if question == "0":
                print("Usubiye ku rupapuro rw'ibanze...\n")
                break

            # Split context into chunks
            chunks = chunk_text(context)
            best_answer = None
            best_score = 0

            # Evaluate each chunk and select the answer with highest score
            for chunk in chunks:
                try:
                    result = qa_pipeline(question=question, context=chunk)
                    if result["score"] > best_score:
                        best_answer = result["answer"]
                        best_score = result["score"]
                except:
                    continue  # skip any problematic chunk

            # If no answer is found, give a fallback message
            if best_answer:
                print("Igisubizo:", best_answer, "\n")
            else:
                print("Ntacyasubijwe. Ongera ugerageze ikibazo cyawe.\n")

    elif choice == "2":
        print("Wahisemo gusubira inyuma...\n")
        continue

    elif choice == "3":
        print("Murakoze! Muri gusohoka...")
        break

    else:
        print("Amahitamo ntakwiye, ongera ugerageze.\n")

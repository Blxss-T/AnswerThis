from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json

# Loading the pre-trained model ("mT5 small")
model_name = "xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Q&A dataset
with open("maternal_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Combine all answers into one long text
context = " ".join([item["answer"] for item in qa_data])

# Function to split long text into chunks
def chunk_text(text, max_tokens=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

# Create a pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
print("=========!Murakaza neza kuri MotherLink Baza !==========")

# Run a loop to ask questions
while True:
    choice = input("1. Gukomeza kubaza | 2. Gusubira inyuma | 3. Gusohokamo: ")

    if choice == "1":
        while True:
            question = input("Shyiramo ikibazo (andika '0' kugira usubire inyuma): ")
            if question == "0":
                break

            # Apply chunking
            chunks = chunk_text(context)
            best_answer = None
            best_score = 0

            # Evaluate each chunk
            for chunk in chunks:
                result = qa_pipeline(question=question, context=chunk)
                if result["score"] > best_score:
                    best_answer = result["answer"]
                    best_score = result["score"]

            print("Igisubizo:", best_answer, "\n")

    elif choice == "2":
        print("Wahisemo gusubira inyuma...")
        continue

    elif choice == "3":
        print("Murakoze! Muri gusohoka...")
        break

    else:
        print("Amahitamo ntakwiye, ongera ugerageze.\n")

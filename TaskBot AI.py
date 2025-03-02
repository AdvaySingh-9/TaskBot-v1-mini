from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from fastapi import FastAPI
# Use Google's Flan-T5-Base model from Hugging Face
model_id = "google/flan-t5-base"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

app = Flask(__name__)

# Load abusive words list
with open("abusive_words.json", "r") as f:
    abusive_words = json.load(f)

safe_default_message = "An error occurred, please try again later. error:-█"

# Load gods information
with open("gods.json", "r") as f:
    gods = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

def clean_answer(answer):
    # If answer is a list, join its elements into a string separated by newlines
    if isinstance(answer, list):
        answer = "\n".join(answer)
    # Check for any banned word in the answer (case-insensitive)
    for word in abusive_words:
        if word.lower() in answer.lower():
            return safe_default_message
    return answer

@app.route("/ask", methods=["POST"])
def ask():
    # Get the question from the form data
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"Error": "Please give me a question to ask."}), 400
    lower_question = question.lower()

    # Check if the question mentions any god from the JSON
    for god in gods["gods"]:
        # Check primary name and aliases (if present)
        names_to_check = [god["name"].lower()]
        if "aliases" in god:
            names_to_check.extend(alias.lower() for alias in god["aliases"])
        if any(name in lower_question for name in names_to_check):
            answer = god["description"]
            return jsonify({"answer": clean_answer(answer)})

    # Fallback: Use the model to generate an answer if no god is detected
    prompt = (
        "You are an AI created by Advay Singh and Astrumix. "
        "Answer the following question accurately and politely:\n"
        f"Question: {question}\nAnswer: "
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=200, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Answer: {answer}\n--------------------")
    return jsonify({"answer": clean_answer(answer)})

if __name__ == "__main__":
    # Bind to all network interfaces so the app is externally accessible
    app.run(host="0.0.0.0", port=7860)

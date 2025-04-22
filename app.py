from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import os

# Set Hugging Face cache to a writable location
os.environ["HF_HOME"] = "/app/hf_cache"

app = Flask(__name__)

# Try loading abusive words safely
try:
    with open("abusive_words (unsafe to open).json", "r") as f:
        abusive_words = json.load(f)
except FileNotFoundError:
    abusive_words = []
    print("Warning: abusive_words.json not found.")

# Try loading gods data safely
try:
    with open("gods.json", "r") as f:
        gods = json.load(f)
except FileNotFoundError:
    gods = {"gods": []}
    print("Warning: gods.json not found.")

safe_default_message = "An error occurred, please try again later. error:-█"

# Load Google's Flan-T5-Base model
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cpu")  # Use CPU to avoid memory overload

@app.route("/")
def index():
    return render_template("index.html")

def clean_answer(answer):
    """Checks for abusive words and returns a safe response."""
    if isinstance(answer, list):
        answer = "\n".join(answer)
    for word in abusive_words:
        if word.lower() in answer.lower():
            return safe_default_message
    return answer

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    lower_question = question.lower()

    # Check if the question mentions any god
    for god in gods.get("gods", []):
        names_to_check = [god["name"].lower()]
        names_to_check.extend(alias.lower() for alias in god.get("aliases", []))
        if any(name in lower_question for name in names_to_check):
            return jsonify({"answer": clean_answer(god["description"])})

    print(f"Question: {question}")
    
    # Generate answer with the model
    prompt = (
        "You are an AI created by Advay Singh and Astrumix. "
        "Answer the following question accurately and politely:\n"
        f"Question: {question}\nAnswer: "
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=150, early_stopping=True)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Answer: {answer}\n--------------------")

    return jsonify({"answer": clean_answer(answer)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

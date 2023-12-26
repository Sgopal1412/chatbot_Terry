from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_chat_response():
    user_input = request.form.get("msg")
    
    # Generate chat response using DialoGPT model
    chat_history_ids = model.generate(
        tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt'),
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    
    bot_response = tokenizer.decode(
        chat_history_ids[:, tokenizer.encode(user_input, return_tensors='pt').shape[-1]:][0],
        skip_special_tokens=True
    )

    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run()

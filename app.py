
from flask import Flask, render_template, request
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from pyngrok import ngrok
import os
import torch

# إغلاق أي أنفاق نشطة
ngrok.kill()

# ضع التوكن الذي حصلت عليه من ngrok
ngrok.set_auth_token("2qoZ1dcMbiUE6wFC4c8GYvnQgI2_3oaF296YozKoVhUmp8m33")

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج وTokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# تعيين pad_token_id يدويًا إذا لم يكن موجودًا
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(prompt):
    decoded_response = ""
    if "how many countries are there in asia" in prompt.lower():
        decoded_response = "There are 49 countries in Asia."
    else:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).long()
        response = model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
        decoded_response = decoded_response.split('.')[0] + "."
    print(f"Generated response: {decoded_response}")
    return decoded_response

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')

# صفحة الشات
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    print(f"User input: {user_input}")
    response = generate_response(user_input)
    print(f"Response sent to client: {response}")
    return render_template('index.html', response=response)

# استخدام ngrok لفتح القناة
try:
    public_url = ngrok.connect(5000)
    print('تم فتح القناة على: ' + public_url.public_url)
except Exception as e:
    print(f"حدث خطأ أثناء فتح القناة: {e}")

# تشغيل الخادم Flask
app.run(host="0.0.0.0", port=5000)

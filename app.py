import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from flask import Flask, render_template, request
model = BartForConditionalGeneration.from_pretrained(r'D:\projects\Abstractive Summarization Project Data\model')
tokenizer = BartTokenizer.from_pretrained(r'D:\projects\Abstractive Summarization Project Data\model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

app = Flask(__name__)

@app.route('/')


@app.route('/res', methods=['GET', 'POST'])
def index():
    
    textarea_data =""
    output_data=""
    if request.method == 'POST':
    
        textarea_data = request.form['input']
        input_ids = tokenizer.encode(textarea_data, return_tensors='pt').to(device)
        summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
        output_data = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", input_data=textarea_data, output_data=output_data)

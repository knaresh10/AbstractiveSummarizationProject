import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup
import json

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

train_dataset = [
    {
        "input_text" : "Perseus “Percy” Jackson is the main protagonist and the narrator of the Percy Jackson and the Olympians series.",
        "summary_text" : "Percy is the protagonist of Percy Jackson and the Olympians",
    },
    {
        "input_text" : "Annabeth Chase is one of the main protagonists in Percy Jackson and the Olympians.",
        "summary_text" : "Annabeth is a protagonist in Percy Jackson and the Olympians.",
    },
]

epochs = 1
batch_size = 8
learning_rate = 2e-5
warmup_steps = 500
total_steps = len(train_dataset) * epochs // batch_size


def collate_fn(batch):
    input_texts = [example['input_text'] for example in batch]
    summary_texts = [example['summary_text'] for example in batch]
    input_ids = tokenizer.batch_encode_plus(input_texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
    summary_ids = tokenizer.batch_encode_plus(summary_texts, padding=True, truncation=True, return_tensors='pt')['input_ids']
    return {'input_ids': input_ids, 'attention_mask': input_ids.ne(tokenizer.pad_token_id), 'labels': summary_ids}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
       
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
       
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
   
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


model.save_pretrained(r'D:\projects\Abstractive Summarization Project Data\model')
tokenizer.save_pretrained(r'D:\projects\Abstractive Summarization Project Data\model')


training_parameters = {
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'warmup_steps': warmup_steps,
    'total_steps': total_steps,
}
with open(r'D:\projects\Abstractive Summarization Project Data\model\training_parameters.json', 'w') as f:
    json.dump(training_parameters, f)

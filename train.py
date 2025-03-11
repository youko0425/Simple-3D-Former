from transformers import ViTForImageClassification

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)

def train_model(model, data_loader, optimizer, loss_fn):
    model.train()
    for data in data_loader:
        inputs = data['inputs']
        labels = data['labels']
        
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

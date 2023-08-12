import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        #print(data)
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn1(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            logits, loss = model(**data)
            final_loss += loss.item()

            # Assuming your model outputs predicted labels (logits)
            predicted_labels = logits.argmax(dim=-1)
            true_labels = data['labels']  # Replace with the actual key for true labels

            correct_preds += (predicted_labels == true_labels).sum().item()
            total_preds += true_labels.numel()

    accuracy = correct_preds / total_preds
    avg_loss = final_loss / len(data_loader)
    return avg_loss, accuracy

import torch
from model import AdvancedTransformer
from data_loader import get_data_loader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

def evaluate_model(device):
    model = AdvancedTransformer().model.to(device)
    data_loader = get_data_loader(train=False)

    model.eval()
    all_predictions, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Log evaluation loss to TensorBoard
            writer.add_scalar("Loss/eval", loss.item(), step)

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    writer.add_scalar("Accuracy/eval", accuracy)
    writer.close()

    return accuracy

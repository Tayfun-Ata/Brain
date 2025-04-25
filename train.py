import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from model import AdvancedTransformer
from data_loader import get_data_loader

def train_model(device):
    model = AdvancedTransformer().model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    data_loader = get_data_loader()
    num_training_steps = len(data_loader) * 3  # Example: 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # TensorBoard writer
    writer = SummaryWriter()

    model.train()
    step = 0  # Initialize step counter
    try:
        for epoch in range(3):  # Example: 3 epochs
            for batch in data_loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with autocast(device_type="cpu"):
                    outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # Log loss to TensorBoard
                writer.add_scalar("Loss/train", loss.item(), step)
                step += 1  # Increment step counter

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")
    finally:
        writer.close()
        print("Training stopped.")

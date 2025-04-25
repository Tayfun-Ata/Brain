import torch
from train import train_model
from evaluate import evaluate_model
from data_collector import start_data_collection

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start data collection in the background
    print("Starting data collection...")
    start_data_collection()

    # Early stopping parameters
    no_improvement_cycles = 0
    max_no_improvement_cycles = 5  # Stop after 5 cycles without improvement
    best_accuracy = 0.0

    # Continuous training and evaluation loop
    while True:
        # Train the model
        train_model(device=device)

        # Evaluate and improve the model
        accuracy = evaluate_model(device=device)

        # Check for improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_cycles = 0
            print(f"New best accuracy: {best_accuracy:.4f}. Model saved.")
        else:
            no_improvement_cycles += 1
            print(f"No improvement. Cycles without improvement: {no_improvement_cycles}")

        # Stop training if no improvement for a specified number of cycles
        if no_improvement_cycles >= max_no_improvement_cycles:
            print("Early stopping triggered. Training stopped.")
            break

        print("Cycle complete. Continuing training...")

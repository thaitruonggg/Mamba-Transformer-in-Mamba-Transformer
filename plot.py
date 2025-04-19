import re
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_training_metrics(log_file_path):
    """
    Read training log file, parse it, and plot accuracy vs epoch and loss vs epoch
    as separate files.

    Args:
        log_file_path (str): Path to the file containing training logs
    """
    # Read the log file
    try:
        with open(log_file_path, 'r') as file:
            log_text = file.read()
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract epoch, test loss, and accuracy data using regular expressions
    epoch_data = re.findall(r'Epoch \[(\d+)/\d+\], Test Loss: (\d+\.\d+), Overall Accuracy: (\d+\.\d+)%', log_text)
    highest_acc = re.findall(r'Highest accuracy: (\d+\.\d+)% achieved at epoch (\d+)', log_text)

    if not epoch_data:
        print("No training data found in the log file. Check the file format.")
        return

    # Extract training loss data for each epoch
    train_loss_data = re.findall(r'Epoch \[(\d+)/\d+\], Iter \[\d+/\d+\], Loss: (\d+\.\d+)', log_text)

    epochs = []
    test_losses = []
    accuracies = []

    # Process test loss and accuracy data
    for epoch, loss, acc in epoch_data:
        epochs.append(int(epoch))
        test_losses.append(float(loss))
        accuracies.append(float(acc))

    # Create a dictionary to store training losses by epoch
    train_losses_by_epoch = {}
    for epoch, loss in train_loss_data:
        epoch = int(epoch)
        loss = float(loss)
        if epoch not in train_losses_by_epoch:
            train_losses_by_epoch[epoch] = []
        train_losses_by_epoch[epoch].append(loss)

    # Calculate average training loss per epoch
    avg_train_losses = []
    for epoch in epochs:
        if epoch in train_losses_by_epoch:
            avg_train_losses.append(np.mean(train_losses_by_epoch[epoch]))
        else:
            avg_train_losses.append(None)

    # PLOT 1: Accuracy vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracies, 'b-', marker='o', markersize=4, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Epoch', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add horizontal line at highest accuracy
    if highest_acc:
        highest_acc_value = float(highest_acc[-1][0])
        highest_acc_epoch = int(highest_acc[-1][1])
        plt.axhline(y=highest_acc_value, color='r', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('accuracy_vs_epoch.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # PLOT 2: Loss vs epoch
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, test_losses, 'r-', marker='o', markersize=4, linewidth=2, label='Test Loss')
    plt.plot(epochs, avg_train_losses, 'g-', marker='s', markersize=4, linewidth=2, label='Avg Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Epoch', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Set y-axis to logarithmic scale for better visualization of loss values
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('loss_vs_epoch.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Training summary:")
    print(f"  - Total epochs: {epochs[-1]}")
    print(f"  - Final test accuracy: {accuracies[-1]}%")
    print(f"  - Final test loss: {test_losses[-1]}")
    if highest_acc:
        print(f"  - Highest test accuracy: {highest_acc_value}% achieved at epoch {highest_acc_epoch}")

    print(f"\nPlots saved as:")
    print(f"  - 'accuracy_vs_epoch.png'")
    print(f"  - 'loss_vs_epoch.png'")


if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    else:
        # Default file path if not provided
        log_file_path = "re.txt"
        print(f"No file path provided. Using default: {log_file_path}")

    plot_training_metrics(log_file_path)
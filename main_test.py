import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks import PGD, FGSM
import torchsummary
from torchsummary import summary
import shutil
from ptflops import get_model_complexity_info
import warnings
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets

warnings.filterwarnings("ignore")
import torch

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
print("------------------------------------------------")


# Imported from train.py - Dataset preparation function
def dataloader_prepare(train_data_folder_path, test_data_folder_path, batchsize):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize([224, 224]),  # Changed from 32x32 to 224x224 to match main_test.py
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(0.8), contrast=(1, 1)),
        transforms.Resize([224, 224]),  # Changed from 32x32 to 224x224 to match main_test.py
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = datasets.ImageFolder(train_data_folder_path, transform=transform_train)
    test_data = datasets.ImageFolder(test_data_folder_path, transform=transform_val)

    # Dataset lengths
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("Training dataset length: {}".format(train_data_size))
    print("Testing dataset length: {}".format(test_data_size))

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=True)

    return train_dataloader, test_dataloader


def evaluate_model(model, test_loader, criterion, classes, batch_size, epoch, num_epochs, train_on_gpu=True,
                   display_per_class=False):
    test_loss = 0.0
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(correct_tensor.numpy())

            current_batch_size = data.size(0)
            for i in range(current_batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

    # Print summary
    # Only display per-class accuracy if display_per_class is True
    if display_per_class:
        print("Per-Class Accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100. * class_correct[i] / class_total[i]
                print(f'Class {i} ({classes[i]}): {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
            else:
                print(f'Class {i} ({classes[i]}): No samples')

    print("--------------------------------------------------------------------")
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.6f}, Overall Accuracy: {overall_accuracy:.2f}%')

    model.train()  # Switch back to training mode
    return test_loss, overall_accuracy


def track_highest_accuracy(accuracy_list):
    """
    Tracks the highest accuracy from a list of accuracy values.

    Args:
        accuracy_list: List of accuracy values collected during training

    Returns:
        tuple: (highest_accuracy, epoch_with_highest_accuracy)
    """
    highest_accuracy = max(accuracy_list) if accuracy_list else 0.0
    epoch_with_highest = accuracy_list.index(highest_accuracy) + 1 if accuracy_list else 0

    print(f"Highest accuracy: {highest_accuracy:.2f}% achieved at epoch {epoch_with_highest}")
    return highest_accuracy, epoch_with_highest


def plot_training_progress(train_loss_list, test_loss_list, accuracy_list, model_name):
    """
    Plot training progress showing accuracy vs epoch and loss vs epoch

    Args:
        train_loss_list: List of training loss values per epoch
        test_loss_list: List of test loss values per epoch
        accuracy_list: List of accuracy values per epoch
        model_name: Name of the model for plot titles
    """
    epochs = range(1, len(accuracy_list) + 1)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracy vs epoch
    ax1.plot(epochs, accuracy_list, 'o-', color='blue', label='Test Accuracy')
    ax1.set_title(f'{model_name} - Accuracy vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot loss vs epoch
    ax2.plot(epochs, train_loss_list, 'o-', color='red', label='Training Loss')
    ax2.plot(epochs, test_loss_list, 'o-', color='green', label='Test Loss')
    ax2.set_title(f'{model_name} - Loss vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_progress.png')
    plt.show()


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, normalize=True):
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 20))

    for i in range(rows * cols):
        if i >= len(images):
            break

        ax = fig.add_subplot(rows, cols, i + 1)
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_fps(model):
    # Integrated from train.py
    iterations = 300  # Repeat count

    device = torch.device("cuda:0")
    model = model.to(device)

    random_input = torch.randn(1, 3, 224, 224).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU warm-up
    for _ in range(50):
        _ = model(random_input)

    # Speed test
    times = torch.zeros(iterations)  # Store times for each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronize GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # Calculate time
            times[iter] = curr_time

    mean_time = times.mean().item()
    print("Inference time: {:.6f} ms, FPS: {:.2f}".format(mean_time, 1000.0 / mean_time))
    inference_time = mean_time
    fps = 1000.0 / mean_time
    return inference_time, fps


# Data organization code (keeping the original from main_test.py)
# Comment out if you're using the dataloader_prepare function directly
"""
# GTSRB
# Organize test data
data_dir = 'GTSRB/GTSRB_Final_Test_Images/GTSRB'
images_dir = os.path.join(data_dir, 'Final_Test/Images')
test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

with open('GTSRB/GTSRB_Final_Test_GT/GT-final_test.csv') as f:
    image_names = f.readlines()

for text in image_names[1:]:
    classes = int(text.split(';')[-1])
    image_name = text.split(';')[0]

    test_class_dir = os.path.join(test_dir, f"{classes:04d}")
    os.makedirs(test_class_dir, exist_ok=True)
    image_path = os.path.join(images_dir, image_name)

    shutil.copy(image_path, test_class_dir)
"""

if __name__ == "__main__":
    # Define paths and parameters
    batch_size = 50
    train_data_path = 'GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
    test_data_path = 'GTSRB/GTSRB_Final_Test_Images/GTSRB/test'

    # Use the integrated dataloader_prepare function
    train_loader, test_loader = dataloader_prepare(train_data_path, test_data_path, batch_size)

    # Get classes from train data
    trainset = torchvision.datasets.ImageFolder(root=train_data_path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),
                                                    transforms.ToTensor(),
                                                ]))

    testset = torchvision.datasets.ImageFolder(root=test_data_path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                               ]))

    # Visualize a batch of images
    batch = next(iter(train_loader))
    classes = trainset.classes
    plot_images(batch[0], batch[1], classes)

    # Load and modify model
    from LNL_test import LNL_Ti as small

    model = small(pretrained=False)
    model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
    model = model.cuda()

    # Train Locality-iN-Locality
    num_epochs = 100
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    lnl_accuracy_list = []
    lnl_train_loss_list = []
    lnl_test_loss_list = []

    for epoch in range(num_epochs):
        total_batch = len(trainset) // batch_size
        running_loss = 0.0

        model.train()
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            X = batch_images.cuda()
            Y = batch_labels.cuda()

            pre = model(X)
            cost = loss(pre, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            running_loss += cost.item()

            if (i + 1) % 200 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.6f' %
                      (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        lnl_train_loss_list.append(avg_train_loss)

        # Add evaluation after each epoch (without per-class accuracy)
        test_loss, test_accuracy = evaluate_model(
            model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False)
        lnl_test_loss_list.append(test_loss)
        lnl_accuracy_list.append(test_accuracy)
        highest_acc, best_epoch = track_highest_accuracy(lnl_accuracy_list)

    print("--------------------------------------------------------------------")
    print("Final Evaluation of Locality-iN-Locality Model")
    # Final evaluation with per-class accuracy
    final_loss, final_accuracy = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, num_epochs - 1, num_epochs, display_per_class=True)
    highest_acc, best_epoch = track_highest_accuracy(lnl_accuracy_list)
    print("--------------------------------------------------------------------")

    # Plot training progress for LNL model
    plot_training_progress(lnl_train_loss_list, lnl_test_loss_list, lnl_accuracy_list, "MiM")

    # Calculate FPS and inference time (added from train.py)
    inference_time, fps = get_fps(model)
    print(f"Model inference metrics - Time: {inference_time:.4f}ms, FPS: {fps:.2f}")

    torch.cuda.empty_cache()

    # Train with MoEx
    from LNL_MoEx_test import LNL_MoEx_Ti as small
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Initialize model
    model = small(pretrained=False)
    model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)  # 43 classes for GTSRB
    model = model.cuda()

    # Hyperparameters
    num_epochs = 100
    moex_lam = .9
    moex_prob = .7

    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    moex_accuracy_list = []
    moex_train_loss_list = []
    moex_test_loss_list = []

    for epoch in range(num_epochs):
        total_batch = len(trainset) // batch_size
        running_loss = 0.0

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()

            prob = torch.rand(1).item()
            if prob < moex_prob:
                swap_index = torch.randperm(input.size(0), device=input.device)
                with torch.no_grad():
                    target_a = target
                    target_b = target[swap_index]
                output = model(input, swap_index=swap_index, moex_norm='pono', moex_epsilon=1e-5,
                               moex_layer='stem', moex_positive_only=False)
                lam = moex_lam
                cost = loss(output, target_a) * lam + loss(output, target_b) * (1. - lam)
            else:
                # compute output
                output = model(input)
                cost = loss(output, target)

            # Backpropagation
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            running_loss += cost.item()

            if (i + 1) % 200 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.6f' %
                      (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        moex_train_loss_list.append(avg_train_loss)

        # Add evaluation after each epoch (without per-class accuracy)
        test_loss, test_accuracy = evaluate_model(
            model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False
        )

        moex_test_loss_list.append(test_loss)
        moex_accuracy_list.append(test_accuracy)
        moex_highest_acc, moex_best_epoch = track_highest_accuracy(moex_accuracy_list)

    print("--------------------------------------------------------------------")
    print("After applying MoEx")
    print("Final Evaluation of LNL-MoEx Model")
    # Final evaluation with per-class accuracy
    final_loss, final_accuracy = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, num_epochs - 1, num_epochs, display_per_class=True)
    moex_highest_acc, moex_best_epoch = track_highest_accuracy(moex_accuracy_list)
    print("--------------------------------------------------------------------")

    # Calculate FPS and inference time for MoEx model (added from train.py)
    inference_time, fps = get_fps(model)
    print(f"MoEx model inference metrics - Time: {inference_time:.4f}ms, FPS: {fps:.2f}")

    # Plot training progress for LNL-MoEx model
    plot_training_progress(moex_train_loss_list, moex_test_loss_list, moex_accuracy_list, "MiM-MoEx")

    # Plot comparison between the two models
    plt.figure(figsize=(15, 6))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, lnl_accuracy_list, 'b-', label='LNL')
    plt.plot(epochs, moex_accuracy_list, 'r-', label='LNL-MoEx')
    plt.title('Model Comparison - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Test Loss comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lnl_test_loss_list, 'b-', label='LNL')
    plt.plot(epochs, moex_test_loss_list, 'r-', label='LNL-MoEx')
    plt.title('Model Comparison - Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    torch.cuda.empty_cache()

    # Model complexity
    with torch.cuda.device(0):
        net = model
        macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                                 verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
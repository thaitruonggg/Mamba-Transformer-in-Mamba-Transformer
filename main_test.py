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
import json
from torch.utils.data import Dataset

torch.cuda.empty_cache()

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
print("------------------------------------------------")


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
    #plt.show()

# TT100K
class TT100KDataset(Dataset):
    def __init__(self, data_dir, annotation_file, split='train', transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            annotation_file (string): Path to the annotation file.
            split (string): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Load annotation file
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Get all traffic sign types
        self.classes = self.annotations['types']

        # Create a mapping from class name to index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Get images that belong to the specified split
        self.images = []
        self.labels = []

        for img_id, img_info in self.annotations['imgs'].items():
            img_path = img_info['path']
            # Check if this image belongs to the requested split
            if split in img_path:
                # For each image, get all objects and their categories
                for obj in img_info['objects']:
                    category = obj['category']
                    # We only care about images that have traffic signs
                    if category in self.classes:
                        bbox = obj['bbox']
                        self.images.append({
                            'img_id': img_id,
                            'path': os.path.join(data_dir, img_path),
                            'bbox': bbox
                        })
                        self.labels.append(self.class_to_idx[category])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = img_info['path']
        bbox = img_info['bbox']

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Crop the traffic sign using the bounding box
        x_min, y_min, x_max, y_max = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        # Ensure the coordinates are within bounds and are integers
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(image.width, int(x_max)), min(image.height, int(y_max))

        if x_min >= x_max or y_min >= y_max:
            # If the bounding box is invalid, use the whole image
            cropped_image = image
        else:
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

        # Apply transformations
        if self.transform:
            cropped_image = self.transform(cropped_image)

        label = self.labels[idx]

        return cropped_image, label


# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define paths
data_dir = 'tt100k_2021'  # Base directory for TT100K dataset
annotation_file = os.path.join(data_dir, 'annotations_all.json')

# Create datasets
trainset = TT100KDataset(
    data_dir=data_dir,
    annotation_file=annotation_file,
    split='train',
    transform=transform
)

testset = TT100KDataset(
    data_dir=data_dir,
    annotation_file=annotation_file,
    split='test',
    transform=transform
)

# Create data loaders
batch_size = 50

train_loader = DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=False
)

# Get classes from the dataset
classes = trainset.classes
print(f"Number of classes: {len(classes)}")
print(f"Training set size: {len(trainset)}")
print(f"Test set size: {len(testset)}")
print("--------------------------------------------------------------------")

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

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')

batch = next(iter(train_loader))
classes = trainset.classes
plot_images(batch[0], batch[1], classes)

# Load and modify model
from LNL import LNL_Ti as small

model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=232, bias=True)
model = model.cuda()

# Train Locality-iN-Locality
num_epochs = 200
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
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

    # Step the scheduler
    # scheduler.step()

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

torch.cuda.empty_cache()

# Train with MoEx
from LNL_MoEx import LNL_MoEx_Ti as small
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=232, bias=True)  # 43 classes for GTSRB
model = model.cuda()

# Hyperparameters
num_epochs = 200
moex_lam = .9
moex_prob = .7

# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
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
#plt.show()

torch.cuda.empty_cache()

# Model complexity
with torch.cuda.device(0):
    net = model
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
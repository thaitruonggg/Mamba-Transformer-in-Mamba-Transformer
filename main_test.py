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
warnings.filterwarnings("ignore")
import torch
torch.autograd.set_detect_anomaly(True)
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

# TT100K
tt100k_datadir = "TT100K"
tt100k_annos_file = os.path.join(tt100k_datadir, "annotations_all.json")
tt100k_train_dir = os.path.join(tt100k_datadir, "train")
tt100k_test_dir = os.path.join(tt100k_datadir, "test")

# Load annotations
with open(tt100k_annos_file, 'r') as f:
    tt100k_annos = json.load(f)

# Define directories for organized TT100K data
tt100k_organized_train_dir = os.path.join(tt100k_datadir, "organized_train")
tt100k_organized_test_dir = os.path.join(tt100k_datadir, "organized_test")

# Create directories for each class
sign_types = tt100k_annos['types']
for sign_type in sign_types:
    os.makedirs(os.path.join(tt100k_organized_train_dir, sign_type), exist_ok=True)
    os.makedirs(os.path.join(tt100k_organized_test_dir, sign_type), exist_ok=True)

# Process training images
for img_id, img_anno in tt100k_annos['imgs'].items():
    # Check if it's a training image
    if img_id.startswith('train'):
        img_path = os.path.join(tt100k_train_dir, img_id + '.jpg')

        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue

        # Process each object in the image
        if 'objects' in img_anno:
            for obj in img_anno['objects']:
                sign_type = obj['category']

                # Create a destination path
                dest_dir = os.path.join(tt100k_organized_train_dir, sign_type)
                dest_path = os.path.join(dest_dir, img_id + '.jpg')

                # Copy the image to its class directory
                if not os.path.exists(dest_path):
                    shutil.copy(img_path, dest_path)

# Process test images
for img_id, img_anno in tt100k_annos['imgs'].items():
    # Check if it's a test image
    if img_id.startswith('test'):
        img_path = os.path.join(tt100k_test_dir, img_id + '.jpg')

        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue

        # Process each object in the image
        if 'objects' in img_anno:
            for obj in img_anno['objects']:
                sign_type = obj['category']

                # Create a destination path
                dest_dir = os.path.join(tt100k_organized_test_dir, sign_type)
                dest_path = os.path.join(dest_dir, img_id + '.jpg')

                # Copy the image to its class directory
                if not os.path.exists(dest_path):
                    shutil.copy(img_path, dest_path)

# Define batch size
batch_size = 50

# Set up data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets for TT100K
tt100k_trainset = torchvision.datasets.ImageFolder(
    root=tt100k_organized_train_dir,
    transform=data_transforms
)

tt100k_testset = torchvision.datasets.ImageFolder(
    root=tt100k_organized_test_dir,
    transform=data_transforms
)

# Create data loaders for TT100K
tt100k_train_loader = torch.utils.data.DataLoader(
    dataset=tt100k_trainset,
    batch_size=batch_size,
    shuffle=True
)

tt100k_test_loader = torch.utils.data.DataLoader(
    dataset=tt100k_testset,
    batch_size=batch_size,
    shuffle=True
)

print(f"TT100K Training set: {len(tt100k_trainset)} images, {len(tt100k_trainset.classes)} classes")
print(f"TT100K Test set: {len(tt100k_testset)} images, {len(tt100k_testset.classes)} classes")

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
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
model = model.cuda()

# Train Locality-iN-Locality
num_epochs = 100
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

lnl_accuracy_list = []

for epoch in range(num_epochs):
    total_batch = len(trainset) // batch_size

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        X = batch_images.cuda()
        Y = batch_labels.cuda()

        pre = model(X)
        cost = loss(pre, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.6f' %
                  (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

    # Step the scheduler
    #scheduler.step()

    # Add evaluation after each epoch (without per-class accuracy)
    test_loss, test_accuracy = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False)
    lnl_accuracy_list.append(test_accuracy)
    highest_acc, best_epoch = track_highest_accuracy(lnl_accuracy_list)
    print("--------------------------------------------------------------------")

print("Final Evaluation of Locality-iN-Locality Model")
# Final evaluation with per-class accuracy
final_loss, final_accuracy = evaluate_model(
    model, test_loader, loss, testset.classes, batch_size, num_epochs - 1, num_epochs, display_per_class=True)
highest_acc, best_epoch = track_highest_accuracy(lnl_accuracy_list)
print("--------------------------------------------------------------------")

torch.cuda.empty_cache()

# Train with Random Erasing
from LNL_MoEx import LNL_MoEx_Ti as small  # Assuming this imports the modified TNT
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)  # 43 classes for GTSRB
model = model.cuda()

# Hyperparameters
num_epochs = 100
erase_prob = 0.5  # Probability of applying Random Erasing (replaces moex_prob)

# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

moex_accuracy_list = []

# Assuming train_loader is defined elsewhere
for epoch in range(num_epochs):
    total_batch = len(trainset) // batch_size

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        # Randomly decide whether to apply Random Erasing
        prob = torch.rand(1).item()
        if prob < erase_prob:
            output = model(input, apply_erasing=True)  # Apply Random Erasing
        else:
            output = model(input, apply_erasing=False)  # No augmentation

        # Compute loss (no Mixup blending needed)
        cost = loss(output, target)

        # Backpropagation
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.6f' %
                  (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

    # Step the scheduler
    #scheduler.step()

    # Add evaluation after each epoch (without per-class accuracy)
    test_loss, test_accuracy = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False
    )
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

torch.cuda.empty_cache()

# Model complexity
with torch.cuda.device(0):
    net = model
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
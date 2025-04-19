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
import json
from torch.cuda.amp import autocast, GradScaler

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
from PIL import Image
from torch.utils.data import Dataset

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
# TT100K dataset paths
tt100k_datadir = "tt100k_2021"  # Adjust this to your actual data path
tt100k_annos_file = os.path.join(tt100k_datadir, "annotations_all.json")
tt100k_train_dir = os.path.join(tt100k_datadir, "train")
tt100k_test_dir = os.path.join(tt100k_datadir, "test")

# Load annotations with error handling
with open(tt100k_annos_file, 'r') as f:
    tt100k_annos = json.load(f)
print(
    f"Successfully loaded annotations with {len(tt100k_annos['imgs'])} images and {len(tt100k_annos['types'])} sign types")


# Create a custom dataset class for TT100K with better error handling
class TT100KDataset(Dataset):
    def __init__(self, data_dir, annos, split='train', transform=None):
        self.data_dir = data_dir
        self.annos = annos
        self.split = split
        self.transform = transform

        # Initialize these attributes first
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(annos['types']))}

        # Get a list of all files in the directory
        split_dir = os.path.join(self.data_dir, split)
        all_files = set([f.split('.')[0] for f in os.listdir(split_dir) if f.endswith('.jpg')])

        # Match annotations with files
        valid_images_found = 0
        for img_id, img_anno in annos['imgs'].items():
            if img_id in all_files:
                if 'objects' in img_anno and len(img_anno['objects']) > 0:
                    for obj in img_anno['objects']:
                        sign_type = obj['category']
                        if sign_type in self.class_to_idx:
                            img_path = os.path.join(split_dir, f"{img_id}.jpg")
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[sign_type])
                            valid_images_found += 1
                            break  # Only use the first valid sign in each image

        # Now that self.labels is populated, we can check max label
        if self.labels:  # Only check if labels exist
            max_label = max(self.labels)
            num_classes = len(self.class_to_idx)
            if max_label >= num_classes:
                print("WARNING: Max label index exceeds number of classes!")

        if valid_images_found == 0:
            print("WARNING: No valid images found. Trying alternative approach...")
            # Alternative approach: use all images and assign dummy labels
            self.image_paths = [os.path.join(split_dir, f) for f in os.listdir(split_dir)
                                if f.endswith('.jpg')]
            self.labels = [0] * len(self.image_paths)  # Assign all to first class as fallback
            print(f"Alternative approach found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        return list(self.class_to_idx.keys())


# Define transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create TT100K datasets
trainset = TT100KDataset(
    data_dir=tt100k_datadir,
    annos=tt100k_annos,
    split='train',
    transform=data_transforms
)

testset = TT100KDataset(
    data_dir=tt100k_datadir,
    annos=tt100k_annos,
    split='test',
    transform=data_transforms
)

# Define batch size
batch_size = 32

train_loader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)


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
    # scheduler.step()

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
from LNL_MoEx import LNL_MoEx_Ti as small

model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=232, bias=True)
model = model.cuda()

num_epochs = 400
moex_lam = .9
moex_prob = .7

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

moex_accuracy_list = []

for epoch in range(num_epochs):

    total_batch = len(trainset) // batch_size

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
            # if args.prof >= 0: torch.cuda.nvtx.range_pop()
            cost = loss(output, target)

        # Backpropagation
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.6f' %
                  (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

    # Step the scheduler
    # scheduler.step()

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

with torch.cuda.device(0):
    net = model
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
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
        print("\nPer-Class Accuracy:")
        for i in range(num_classes):
            if class_total[i] > 0:
                accuracy = 100. * class_correct[i] / class_total[i]
                print(f'Class {i} ({classes[i]}): {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
            else:
                print(f'Class {i} ({classes[i]}): No samples')

    print("--------------------------------------------------------------------")
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.6f}, Overall Accuracy: {overall_accuracy:.2f}%')
    print("--------------------------------------------------------------------")

    model.train()  # Switch back to training mode
    return test_loss, overall_accuracy

#GTSRB
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

batch_size = 50

trainset = torchvision.datasets.ImageFolder(root='GTSRB/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images',
                                                transform=transforms.Compose([
                                                          transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          ]),
                                               )

testset = torchvision.datasets.ImageFolder(root='GTSRB/GTSRB_Final_Test_Images/GTSRB/test',
                                                transform=transforms.Compose([
                                                          transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          ]),
                                               )

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )
'''
# GTSRB class names (43 classes)
gtsrb_class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
    "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5t", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5t prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5t"
]

# Function to count samples per class
def get_class_distribution(dataset, name="Dataset"):
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts
# Get class distributions
train_class_counts = get_class_distribution(trainset, "Training Set")
test_class_counts = get_class_distribution(testset, "Test Set")
# Prepare data for plotting
num_classes = len(gtsrb_class_names)  # Should be 43
train_counts = [train_class_counts.get(i, 0) for i in range(num_classes)]
test_counts = [test_class_counts.get(i, 0) for i in range(num_classes)]
# Plotting
fig, ax = plt.subplots(figsize=(15, 8))  # Increased height to accommodate vertical labels
x = np.arange(num_classes)
width = 0.35  # Width of the bars
# Plot bars for train and test sets
ax.bar(x - width/2, train_counts, width, label='Training Set', color='skyblue')
ax.bar(x + width/2, test_counts, width, label='Test Set', color='salmon')
# Customize the plot
ax.set_xlabel('Traffic Sign Class')
ax.set_ylabel('Number of Images')
ax.set_title('Number of Images per Class in GTSRB Dataset')
ax.set_xticks(x)
ax.set_xticklabels(gtsrb_class_names, rotation=90, ha='center', fontsize=8)  # Vertical labels
ax.legend()
# Adjust layout to prevent label cutoff
plt.tight_layout()
# Display the plot
#plt.show()
'''

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


print("Final Evaluation of Locality-iN-Locality Model")
# Final evaluation with per-class accuracy
final_loss, final_accuracy = evaluate_model(
    model, test_loader, loss, testset.classes, batch_size, num_epochs - 1, num_epochs, display_per_class=True)

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

print("After applying MoEx")
print("Final Evaluation of LNL-MoEx Model")
# Final evaluation with per-class accuracy
final_loss, final_accuracy = evaluate_model(
    model, test_loader, loss, testset.classes, batch_size, num_epochs - 1, num_epochs, display_per_class=True)

# Model complexity
with torch.cuda.device(0):
    net = model
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
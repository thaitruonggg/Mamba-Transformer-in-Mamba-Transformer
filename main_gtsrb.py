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
from trivial_augment import TrivialAugment

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)
print("------------------------------------------------")


# Function to evaluate model
def evaluate_model(model, test_loader, criterion, classes, batch_size, epoch, num_epochs, train_on_gpu=True,
                   display_per_class=False):
    test_loss = 0.0
    num_classes = len(classes)
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    all_preds_list = []
    all_targets_list = []

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

            _, pred = torch.max(output, 1)

            if train_on_gpu:
                all_preds_list.extend(pred.cpu().numpy())
                all_targets_list.extend(target.cpu().numpy())
            else:
                all_preds_list.extend(pred.numpy())
                all_targets_list.extend(target.numpy())

            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy()) if train_on_gpu else np.squeeze(correct_tensor.numpy())

            current_batch_size = data.size(0)
            for i in range(current_batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

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

    model.train()
    return test_loss, overall_accuracy, all_targets_list, all_preds_list

# Function to track the highest accuracy
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

# Function to plot the training process
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 12))

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
    # plt.show()

def calculate_f1_scores(y_true, y_pred, num_classes, average='macro'):
    """
    Calculates F1 score.
    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        num_classes: The total number of classes.
        average: Type of averaging to perform on the data:
                 'micro': Calculate metrics globally by counting the total true positives,
                          false negatives and false positives. (Note: often equals accuracy in multi-class)
                 'macro': Calculate metrics for each label, and find their unweighted mean.
                          This does not take label imbalance into account.
                 'weighted': Calculate metrics for each label, and find their average,
                             weighted by support (the number of true instances for each label).
                 'per_class': Returns F1 score for each class as a list.

    Returns:
        F1 score (float or list of floats if 'per_class').
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:  # Handles empty list case
        if average == 'per_class':
            return [0.0] * num_classes if num_classes > 0 else []
        return 0.0

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])

        # Assuming labels are 0-indexed and within [0, num_classes-1]
        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    if average == 'micro':
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)
        total_fn = np.sum(fn)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        if (micro_precision + micro_recall) == 0:
            return 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        return micro_f1

    f1_per_class = []
    supports = []

    for c in range(num_classes):
        precision_c = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall_c = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0

        support_c = tp[c] + fn[c]
        supports.append(support_c)

        if (precision_c + recall_c) == 0:
            f1_per_class.append(0.0)
        else:
            f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c)
            f1_per_class.append(f1_c)

    if average == 'per_class':
        return f1_per_class
    elif average == 'macro':
        if not f1_per_class: return 0.0
        return np.mean(f1_per_class)
    elif average == 'weighted':
        if np.sum(supports) == 0:  # no samples means F1 is 0
            return 0.0
        return np.average(f1_per_class, weights=supports)
    else:
        raise ValueError("average parameter must be 'micro', 'macro', 'weighted', or 'per_class'")

def calculate_precision_scores(y_true, y_pred, num_classes, average='macro'):
    """
    Calculates Precision score.

    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        num_classes: The total number of classes.
        average: Type of averaging to perform on the data:
                 'micro': Calculate metrics globally by counting the total true positives,
                          and false positives.
                 'macro': Calculate metrics for each label, and find their unweighted mean.
                          This does not take label imbalance into account.
                 'weighted': Calculate metrics for each label, and find their average,
                             weighted by support (the number of true instances for each label).
                 'per_class': Returns Precision score for each class as a list.

    Returns:
        Precision score (float or list of floats if 'per_class').
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:  # Handles empty list case
        if average == 'per_class':
            return [0.0] * num_classes if num_classes > 0 else []
        return 0.0

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])

        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    if average == 'micro':
        total_tp = np.sum(tp)
        total_fp = np.sum(fp)

        if (total_tp + total_fp) == 0:
            return 0.0
        micro_precision = total_tp / (total_tp + total_fp)
        return micro_precision

    precision_per_class = []
    supports = []

    for c in range(num_classes):
        if (tp[c] + fp[c]) == 0:
            precision_per_class.append(0.0)
        else:
            precision_c = tp[c] / (tp[c] + fp[c])
            precision_per_class.append(precision_c)

        support_c = tp[c] + fn[c]
        supports.append(support_c)

    if average == 'per_class':
        return precision_per_class
    elif average == 'macro':
        if not precision_per_class: return 0.0
        return np.mean(precision_per_class)
    elif average == 'weighted':
        if np.sum(supports) == 0:
            return 0.0
        return np.average(precision_per_class, weights=supports)
    else:
        raise ValueError("average parameter must be 'micro', 'macro', 'weighted', or 'per_class'")

def calculate_recall_scores(y_true, y_pred, num_classes, average='macro'):
    """
    Calculates Recall score.
    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        num_classes: The total number of classes.
        average: Type of averaging to perform on the data:
                 'micro': Calculate metrics globally by counting the total true positives,
                          and false negatives.
                 'macro': Calculate metrics for each label, and find their unweighted mean.
                          This does not take label imbalance into account.
                 'weighted': Calculate metrics for each label, and find their average,
                             weighted by support (the number of true instances for each label).
                 'per_class': Returns Recall score for each class as a list.

    Returns:
        Recall score (float or list of floats if 'per_class').
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        if average == 'per_class':
            return [0.0] * num_classes if num_classes > 0 else []
        return 0.0

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])

        if true_label == pred_label:
            tp[true_label] += 1
        else:
            fp[pred_label] += 1
            fn[true_label] += 1

    if average == 'micro':
        total_tp = np.sum(tp)
        total_fn = np.sum(fn)

        if (total_tp + total_fn) == 0:
            return 0.0
        micro_recall = total_tp / (total_tp + total_fn)
        return micro_recall

    recall_per_class = []
    supports = []

    for c in range(num_classes):
        if (tp[c] + fn[c]) == 0:
            recall_per_class.append(0.0)
        else:
            recall_c = tp[c] / (tp[c] + fn[c])
            recall_per_class.append(recall_c)

        support_c = tp[c] + fn[c]
        supports.append(support_c)

    if average == 'per_class':
        return recall_per_class
    elif average == 'macro':
        if not recall_per_class: return 0.0
        return np.mean(recall_per_class)
    elif average == 'weighted':
        if np.sum(supports) == 0:
            return 0.0
        return np.average(recall_per_class, weights=supports)
    else:
        raise ValueError("average parameter must be 'micro', 'macro', 'weighted', or 'per_class'")

def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model"):
    """
    Computes, prints, and plots the confusion matrix.
    Args:
        y_true: List or array of true labels.
        y_pred: List or array of predicted labels.
        class_names: List of names for each class.
        model_name: Name of the model for the plot title.
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    print(f"\nConfusion Matrix for {model_name}:")

    header = "Pred ->" + " ".join([f"{name[:3]:>5}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:5]:<5} |" + " ".join([f"{val:5}" for val in row])
        print(row_str)
    print("-" * len(header))

    if num_classes > 20:
        fig_size = (20, 20)
        cell_fontsize = 10
        tick_label_fontsize = 10
    elif num_classes > 10:
        fig_size = (18, 18)
        cell_fontsize = 10
        tick_label_fontsize = 12
    else:
        fig_size = (12, 12)
        cell_fontsize = 12
        tick_label_fontsize = 12

    axis_title_fontsize = 14
    plot_title_fontsize = 16

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=tick_label_fontsize)
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_yticklabels(class_names, fontsize=tick_label_fontsize)

    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=plot_title_fontsize)
    ax.set_ylabel('True label', fontsize=axis_title_fontsize)
    ax.set_xlabel('Predicted label', fontsize=axis_title_fontsize)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=cell_fontsize)
    fig.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_confusion_matrix.png')
    # plt.show()

# GTSRB dataset
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
                                                transforms.Resize((224, 224)),
                                                TrivialAugment(),
                                                transforms.ToTensor(),
                                            ]),
                                            )

testset = torchvision.datasets.ImageFolder(root='GTSRB/GTSRB_Final_Test_Images/GTSRB/test',
                                           transform=transforms.Compose([
                                               transforms.Resize((224, 224)),
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

# GTSRB class names
gtsrb_class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
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
    "End of no passing for vehicles over 3.5 metric tons"
]

# Function to count samples per class
def get_class_distribution(dataset, name="Dataset"):
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

train_class_counts = get_class_distribution(trainset, "Training Set")
test_class_counts = get_class_distribution(testset, "Test Set")
num_classes = len(gtsrb_class_names)
train_counts = [train_class_counts.get(i, 0) for i in range(num_classes)]
test_counts = [test_class_counts.get(i, 0) for i in range(num_classes)]

# Plotting
fig, ax = plt.subplots(figsize=(19, 12))
x = np.arange(num_classes)
width = 0.35
ax.bar(x - width / 2, train_counts, width, label='Training Set', color='skyblue')
ax.bar(x + width / 2, test_counts, width, label='Test Set', color='salmon')
# Customize the plot
ax.set_xlabel('Traffic Sign Class')
ax.set_ylabel('Number of Images')
ax.set_title('Number of Images per Class in GTSRB Dataset')
ax.set_xticks(x)
ax.set_xticklabels(gtsrb_class_names, rotation=90, ha='center', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('gtsrb_dataset_distribution.png')
#plt.show()

# Function to normalize and plot image
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

    fig = plt.figure(figsize=(25, 25))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]], fontsize=15)
        ax.axis('off')


batch = next(iter(train_loader))
classes = trainset.classes
plot_images(batch[0], batch[1], classes)
plt.savefig('gtsrb_batch_sample.png')
plt.close()

# Load and modify model
from MaMa import MaMa_Ti as small

# Initialize model
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)  # out_features = 43 classes for GTSRB
model = model.cuda()

# Hyperparameters
num_epochs = 100
# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Early Stopping parameters for MaMa model
patience_mama = 15
best_mama_accuracy = 0.0
mama_epochs_no_improve = 0
best_mama_state_dict = None
best_mama_epoch = 0

# Train Mamba-Transformer in Mamba-Transformer
mama_accuracy_list = []
mama_train_loss_list = []
mama_test_loss_list = []

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
    mama_train_loss_list.append(avg_train_loss)

    # Add evaluation after each epoch
    current_test_loss, current_test_accuracy, _, _ = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False)
    mama_test_loss_list.append(current_test_loss)
    mama_accuracy_list.append(current_test_accuracy)

    # Early stopping check for MaMa model
    if current_test_accuracy > best_mama_accuracy:
        best_mama_accuracy = current_test_accuracy
        mama_epochs_no_improve = 0
        best_mama_state_dict = model.state_dict().copy()  # Save the best model state
        best_mama_epoch = epoch + 1
        print(
            f"INFO (MaMa): New best accuracy: {best_mama_accuracy:.2f}% at epoch {best_mama_epoch}. Saving model state.")
    else:
        mama_epochs_no_improve += 1
        print(
            f"INFO (MaMa): No improvement for {mama_epochs_no_improve} epochs. Best accuracy was {best_mama_accuracy:.2f}% at epoch {best_mama_epoch}.")

    track_highest_accuracy(mama_accuracy_list)
    print("--------------------------------------------------------------------")

    if mama_epochs_no_improve >= patience_mama:
        print(
            f"INFO (MaMa): Early stopping triggered at epoch {epoch + 1} after {patience_mama} epochs of no improvement.")
        break  # Exit the training loop

# After MaMa training loop
if best_mama_state_dict:
    print(
        f"INFO (MaMa): Loading best model from epoch {best_mama_epoch} with accuracy {best_mama_accuracy:.2f}% for final evaluation.")
    model.load_state_dict(best_mama_state_dict)
else:
    print("INFO (MaMa): No improvement observed, or training stopped very early. Using last model state for MaMa.")

print("Final Evaluation of MaMa Model")
# Adjust epoch number for display in evaluate_model if early stopping occurred
final_eval_epoch_mama = best_mama_epoch - 1 if best_mama_state_dict and best_mama_epoch > 0 else epoch
final_loss, final_accuracy, all_targets, all_preds = evaluate_model(
    model, test_loader, loss, testset.classes, batch_size,
    final_eval_epoch_mama,
    num_epochs, display_per_class=True)
# track_highest_accuracy(mama_accuracy_list)
print(
    f"Final accuracy of loaded MaMa model (from epoch {best_mama_epoch if best_mama_state_dict else 'N/A - used last state'}): {final_accuracy:.2f}%")

num_classes_val = len(testset.classes)
macro_f1 = calculate_f1_scores(all_targets, all_preds, num_classes_val, average='macro')
weighted_f1 = calculate_f1_scores(all_targets, all_preds, num_classes_val, average='weighted')
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")

macro_precision = calculate_precision_scores(all_targets, all_preds, num_classes_val, average='macro')
weighted_precision = calculate_precision_scores(all_targets, all_preds, num_classes_val, average='weighted')
print(f"Macro Precision Score: {macro_precision:.4f}")
print(f"Weighted Precision Score: {weighted_precision:.4f}")

macro_recall = calculate_recall_scores(all_targets, all_preds, num_classes_val, average='macro')
weighted_recall = calculate_recall_scores(all_targets, all_preds, num_classes_val, average='weighted')
print(f"Macro Recall Score: {macro_recall:.4f}")
print(f"Weighted Recall Score: {weighted_recall:.4f}")

print("--------------------------------------------------------------------")
plot_training_progress(mama_train_loss_list, mama_test_loss_list, mama_accuracy_list, "MaMa")
plot_confusion_matrix(all_targets, all_preds, gtsrb_class_names, model_name="MaMa")
# Save the MaMa model
model_save_path_mama = "mama_model.pth"
torch.save(model.state_dict(), model_save_path_mama)
torch.cuda.empty_cache()

# Train with MoEx
from MaMa_MoEx import MaMa_MoEx_Ti as small

# Initialize model
model = small(pretrained=False)
model.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)  # out_features = 43 classes for GTSRB
model = model.cuda()

# Hyperparameters
num_epochs = 100
moex_lam = .9
moex_prob = .7
# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Early Stopping parameters for MaMa-MoEx model
patience_moex = 15
best_moex_accuracy = 0.0
moex_epochs_no_improve = 0
best_moex_state_dict = None
best_moex_epoch = 0

# Train Mamba-Transformer in Mamba-Transformer
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

    # Add evaluation after each epoch
    current_moex_test_loss, current_moex_test_accuracy, _, _ = evaluate_model(
        model, test_loader, loss, testset.classes, batch_size, epoch, num_epochs, display_per_class=False
    )

    moex_test_loss_list.append(current_moex_test_loss)
    moex_accuracy_list.append(current_moex_test_accuracy)

    # Early stopping check for MaMa-MoEx model
    if current_moex_test_accuracy > best_moex_accuracy:
        best_moex_accuracy = current_moex_test_accuracy
        moex_epochs_no_improve = 0
        best_moex_state_dict = model.state_dict().copy()  # Save the best model state
        best_moex_epoch = epoch + 1
        print(
            f"INFO (MaMa-MoEx): New best accuracy: {best_moex_accuracy:.2f}% at epoch {best_moex_epoch}. Saving model state.")
    else:
        moex_epochs_no_improve += 1
        print(
            f"INFO (MaMa-MoEx): No improvement for {moex_epochs_no_improve} epochs. Best accuracy was {best_moex_accuracy:.2f}% at epoch {best_moex_epoch}.")

    track_highest_accuracy(moex_accuracy_list)
    print("--------------------------------------------------------------------")

    if moex_epochs_no_improve >= patience_moex:
        print(
            f"INFO (MaMa-MoEx): Early stopping triggered at epoch {epoch + 1} after {patience_moex} epochs of no improvement.")
        break

# After MaMa-MoEx training loop
if best_moex_state_dict:
    print(
        f"INFO (MaMa-MoEx): Loading best model from epoch {best_moex_epoch} with accuracy {best_moex_accuracy:.2f}% for final evaluation.")
    model.load_state_dict(best_moex_state_dict)
else:
    print(
        "INFO (MaMa-MoEx): No improvement observed, or training stopped very early. Using last model state for MaMa-MoEx.")

print("After applying MoEx")
print("Final Evaluation of MaMa-MoEx Model")
# Adjust epoch number for display in evaluate_model if early stopping occurred
final_eval_epoch_moex = best_moex_epoch - 1 if best_moex_state_dict and best_moex_epoch > 0 else epoch
final_loss, final_accuracy, all_targets_moex, all_preds_moex = evaluate_model(
    model, test_loader, loss, testset.classes, batch_size,
    final_eval_epoch_moex,
    num_epochs, display_per_class=True)
# track_highest_accuracy(moex_accuracy_list)
print(
    f"Final accuracy of loaded MaMa-MoEx model (from epoch {best_moex_epoch if best_moex_state_dict else 'N/A - used last state'}): {final_accuracy:.2f}%")

num_classes_val = len(testset.classes)
macro_f1_moex = calculate_f1_scores(all_targets_moex, all_preds_moex, num_classes_val, average='macro')
weighted_f1_moex = calculate_f1_scores(all_targets_moex, all_preds_moex, num_classes_val, average='weighted')
print(f"Macro F1 Score (MaMa-MoEx): {macro_f1_moex:.4f}")
print(f"Weighted F1 Score (MaMa-MoEx): {weighted_f1_moex:.4f}")

macro_precision_moex = calculate_precision_scores(all_targets_moex, all_preds_moex, num_classes_val, average='macro')
weighted_precision_moex = calculate_precision_scores(all_targets_moex, all_preds_moex, num_classes_val,
                                                     average='weighted')
print(f"Macro Precision Score (MaMa-MoEx): {macro_precision_moex:.4f}")
print(f"Weighted Precision Score (MaMa-MoEx): {weighted_precision_moex:.4f}")

macro_recall_moex = calculate_recall_scores(all_targets_moex, all_preds_moex, num_classes_val, average='macro')
weighted_recall_moex = calculate_recall_scores(all_targets_moex, all_preds_moex, num_classes_val, average='weighted')
print(f"Macro Recall Score (MaMa-MoEx): {macro_recall_moex:.4f}")
print(f"Weighted Recall Score (MaMa-MoEx): {weighted_recall_moex:.4f}")

print("--------------------------------------------------------------------")
plot_training_progress(moex_train_loss_list, moex_test_loss_list, moex_accuracy_list, "MaMa-MoEx")
plot_confusion_matrix(all_targets_moex, all_preds_moex, gtsrb_class_names, model_name="MaMa-MoEx")
# Save the MaMa-MoEx model
model_save_path_moex = "mama_moex_model.pth"
torch.save(model.state_dict(), model_save_path_moex)

# Accuracy comparison
plt.subplot(1, 2, 1)

# Define epoch ranges based on actual run lengths for each model
epochs_mama_plot = range(1, len(mama_accuracy_list) + 1)
epochs_moex_plot = range(1, len(moex_accuracy_list) + 1)

plt.plot(epochs_mama_plot, mama_accuracy_list, 'b-', label='MaMa')
plt.plot(epochs_moex_plot, moex_accuracy_list, 'r-', label='MaMa-MoEx')
plt.title('Model Comparison - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Test Loss comparison
plt.subplot(1, 2, 2)
plt.plot(epochs_mama_plot, mama_test_loss_list, 'b-', label='MaMa Test Loss') # Updated label for clarity
plt.plot(epochs_moex_plot, moex_test_loss_list, 'r-', label='MaMa-MoEx Test Loss') # Updated label for clarity
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
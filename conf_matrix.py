import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_confusion_matrix(data=None):
    """
    Draw a diagonal confusion matrix based on classification data.

    Parameters:
    -----------
    data : dict or None
        Dictionary with class indices as keys and tuples of (correct, total) as values.
        If None, uses the example data from the prompt.

    Returns:
    --------
    None (displays a confusion matrix visualization)
    """
    if data is None:
        # Parse the provided data
        data = {
            0: (60, 60),  # Class 0 (0000): 100.00% (60/60)
            1: (719, 720),  # Class 1 (0001): 99.86% (719/720)
            2: (748, 750),  # Class 2 (0002): 99.73% (748/750)
            3: (432, 450),  # Class 3 (0003): 96.00% (432/450)
            4: (655, 660),  # Class 4 (0004): 99.24% (655/660)
            5: (628, 630),  # Class 5 (0005): 99.68% (628/630)
            6: (135, 150),  # Class 6 (0006): 90.00% (135/150)
            7: (450, 450),  # Class 7 (0007): 100.00% (450/450)
            8: (419, 450),  # Class 8 (0008): 93.11% (419/450)
            9: (480, 480),  # Class 9 (0009): 100.00% (480/480)
            10: (659, 660),  # Class 10 (0010): 99.85% (659/660)
            11: (420, 420),  # Class 11 (0011): 100.00% (420/420)
            12: (678, 690),  # Class 12 (0012): 98.26% (678/690)
            13: (718, 720),  # Class 13 (0013): 99.72% (718/720)
            14: (270, 270),  # Class 14 (0014): 100.00% (270/270)
            15: (210, 210),  # Class 15 (0015): 100.00% (210/210)
            16: (150, 150),  # Class 16 (0016): 100.00% (150/150)
            17: (348, 360),  # Class 17 (0017): 96.67% (348/360)
            18: (357, 390),  # Class 18 (0018): 91.54% (357/390)
            19: (59, 60),  # Class 19 (0019): 98.33% (59/60)
            20: (90, 90),  # Class 20 (0020): 100.00% (90/90)
            21: (80, 90),  # Class 21 (0021): 88.89% (80/90)
            22: (110, 120),  # Class 22 (0022): 91.67% (110/120)
            23: (150, 150),  # Class 23 (0023): 100.00% (150/150)
            24: (87, 90),  # Class 24 (0024): 96.67% (87/90)
            25: (463, 480),  # Class 25 (0025): 96.46% (463/480)
            26: (170, 180),  # Class 26 (0026): 94.44% (170/180)
            27: (51, 60),  # Class 27 (0027): 85.00% (51/60)
            28: (150, 150),  # Class 28 (0028): 100.00% (150/150)
            29: (90, 90),  # Class 29 (0029): 100.00% (90/90)
            30: (145, 150),  # Class 30 (0030): 96.67% (145/150)
            31: (270, 270),  # Class 31 (0031): 100.00% (270/270)
            32: (60, 60),  # Class 32 (0032): 100.00% (60/60)
            33: (209, 210),  # Class 33 (0033): 99.52% (209/210)
            34: (120, 120),  # Class 34 (0034): 100.00% (120/120)
            35: (390, 390),  # Class 35 (0035): 100.00% (390/390)
            36: (116, 120),  # Class 36 (0036): 96.67% (116/120)
            37: (60, 60),  # Class 37 (0037): 100.00% (60/60)
            38: (686, 690),  # Class 38 (0038): 99.42% (686/690)
            39: (88, 90),  # Class 39 (0039): 97.78% (88/90)
            40: (86, 90),  # Class 40 (0040): 95.56% (86/90)
            41: (60, 60),  # Class 41 (0041): 100.00% (60/60)
            42: (85, 90),  # Class 42 (0042): 94.44% (85/90)
        }

    # Calculate total number of samples and overall accuracy
    total_correct = sum(correct for correct, _ in data.values())
    total_samples = sum(total for _, total in data.values())
    overall_accuracy = total_correct / total_samples * 100

    num_classes = len(data)

    # Create confusion matrix
    cm = np.zeros((num_classes, num_classes))

    for cls in data:
        correct, total = data[cls]
        # Place correct predictions on diagonal
        cm[cls, cls] = correct

        # Since we don't know how the remaining samples were misclassified,
        # we'll just note that they were incorrect but not specify where they went
        incorrect = total - correct

        # For visualization purposes, we'll distribute the errors evenly
        # Uncomment this code if you want to visualize misclassifications
        # if incorrect > 0:
        #     indices = [i for i in range(num_classes) if i != cls]
        #     for idx in indices:
        #         cm[cls, idx] = incorrect / (num_classes - 1)

    # Create a plot with a size that scales with the number of classes
    plt.figure(figsize=(min(20, num_classes * 0.3 + 5), min(18, num_classes * 0.3 + 4)))

    # Create a heatmap for the confusion matrix
    # Using a logarithmic normalization to see smaller values better
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='.0f',
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))

    plt.title(f'Confusion Matrix (Diagonal Values)\nOverall Accuracy: {overall_accuracy:.2f}%')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()

    # Create a second visualization showing accuracies per class
    plt.figure(figsize=(15, 6))
    accuracies = {cls: correct / total * 100 for cls, (correct, total) in data.items()}

    # Sort classes by accuracy
    sorted_items = sorted(accuracies.items(), key=lambda x: x[1])
    classes, acc_values = zip(*sorted_items)

    # Create color mapping based on accuracy values
    colors = ['red' if a < 95 else 'orange' if a < 98 else 'green' for a in acc_values]

    # Create bar chart
    plt.barh(classes, acc_values, color=colors)
    plt.axvline(x=95, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=98, color='orange', linestyle='-', alpha=0.3)
    plt.axvline(x=100, color='g', linestyle='-', alpha=0.3)
    plt.xlim(90, 101)  # Set x-axis to start at 90% for better visualization
    plt.ylabel('Class')
    plt.xlabel('Accuracy (%)')
    plt.title('Classification Accuracy by Class (Sorted)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Create a summary table
    plt.figure(figsize=(10, 6))
    plt.axis('off')

    # Count classes by accuracy ranges
    perfect = sum(1 for acc in accuracies.values() if acc == 100)
    high = sum(1 for acc in accuracies.values() if 98 <= acc < 100)
    medium = sum(1 for acc in accuracies.values() if 95 <= acc < 98)
    low = sum(1 for acc in accuracies.values() if acc < 95)

    table_data = [
        ["Accuracy Range", "Number of Classes", "Percentage"],
        ["100% (Perfect)", perfect, f"{perfect / num_classes * 100:.1f}%"],
        ["98-99.99%", high, f"{high / num_classes * 100:.1f}%"],
        ["95-97.99%", medium, f"{medium / num_classes * 100:.1f}%"],
        ["<95%", low, f"{low / num_classes * 100:.1f}%"],
        ["Overall", num_classes, f"{overall_accuracy:.2f}%"]
    ]

    table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Classification Performance Summary')

    plt.show()

    return cm


# Execute function with the provided data
if __name__ == "__main__":
    draw_confusion_matrix()
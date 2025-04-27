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
            0: (276, 278),  # Class 0 (pl80): 99.28% (276/278),
            2: (38, 39),  # Class 2 (p6): 97.44% (38/39),
            12: (125, 125),  # Class 12 (p5): 100.00% (125/125),
            13: (39, 39),  # Class 13 (pm55): 100.00% (39/39),
            14: (279, 281),  # Class 14 (pl60): 99.29% (279/281),
            15: (133, 136),  # Class 15 (ip): 97.79% (133/136),
            16: (515, 520),  # Class 16 (p11): 99.04% (515/520),
            19: (130, 134),  # Class 19 (i2r): 97.01% (130/134),
            22: (110, 110),  # Class 22 (p23): 100.00% (110/110),
            39: (46, 46),  # Class 39 (pg): 100.00% (46/46),
            42: (97, 97),  # Class 42 (il80): 100.00% (97/97),
            50: (36, 37),  # Class 50 (ph4): 97.30% (36/37),
            52: (237, 237),  # Class 52 (i4): 100.00% (237/237),
            55: (45, 45),  # Class 55 (pl70): 100.00% (45/45),
            61: (671, 673),  # Class 61 (pne): 99.70% (671/673),
            64: (60, 63),  # Class 64 (ph4.5): 95.24% (60/63),
            65: (68, 69),  # Class 65 (p12): 98.55% (68/69),
            66: (58, 61),  # Class 66 (p3): 95.08% (58/61),
            68: (204, 208),  # Class 68 (pl5): 98.08% (204/208),
            69: (31, 31),  # Class 69 (w13): 100.00% (31/31),
            72: (99, 100),  # Class 72 (i4l): 99.00% (99/100),
            85: (211, 212),  # Class 85 (pl30): 99.53% (211/212),
            110: (95, 98),  # Class 110 (p10): 96.94% (95/98),
            111: (1006, 1007),  # Class 111 (pn): 99.90% (1006/1007),
            118: (63, 63),  # Class 118 (w55): 100.00% (63/63),
            128: (258, 262),  # Class 128 (p26): 98.47% (258/262),
            134: (101, 110),  # Class 134 (p13): 91.82% (101/110),
            135: (63, 63),  # Class 135 (pr40): 100.00% (63/63),
            138: (57, 57),  # Class 138 (pl20): 100.00% (57/57),
            145: (31, 32),  # Class 145 (pm30): 96.88% (31/32),
            148: (450, 455),  # Class 148 (pl40): 98.90% (450/455),
            158: (135, 137),  # Class 158 (i2): 98.54% (135/137),
            162: (85, 87),  # Class 162 (pl120): 97.70% (85/87),
            168: (37, 37),  # Class 168 (w32): 100.00% (37/37),
            170: (42, 43),  # Class 170 (ph5): 97.67% (42/43),
            175: (138, 141),  # Class 175 (il60): 97.87% (138/141),
            176: (124, 125),  # Class 176 (w57): 99.20% (124/125),
            179: (214, 215),  # Class 179 (pl100): 99.53% (214/215),
            183: (62, 63),  # Class 183 (w59): 98.41% (62/63),
            188: (39, 39),  # Class 188 (il100): 100.00% (39/39),
            193: (34, 34),  # Class 193 (p19): 100.00% (34/34),
            216: (46, 49),  # Class 216 (pm20): 93.88% (46/49),
            220: (512, 515),  # Class 220 (i5): 99.42% (512/515),
            223: (47, 47),  # Class 223 (p27): 100.00% (47/47),
            224: (354, 355),  # Class 224 (pl50): 99.72% (354/355)
        }

    # Calculate total number of samples and overall accuracy
    total_correct = sum(correct for correct, _ in data.values())
    total_samples = sum(total for _, total in data.values())
    overall_accuracy = total_correct / total_samples * 100

    # Get the actual class indices from the data dictionary
    class_indices = sorted(data.keys())
    num_classes = len(class_indices)

    # Create a mapping from actual class indices to sequential indices (0 to num_classes-1)
    class_to_idx = {cls: i for i, cls in enumerate(class_indices)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    # Create confusion matrix with the correct dimensions
    cm = np.zeros((num_classes, num_classes))

    for cls in data:
        correct, total = data[cls]
        # Get the sequential index for this class
        seq_idx = class_to_idx[cls]

        # Place correct predictions on diagonal
        cm[seq_idx, seq_idx] = correct

        # Since we don't know how the remaining samples were misclassified,
        # we'll just note that they were incorrect but not specify where they went
        incorrect = total - correct

        # For visualization purposes, we'll distribute the errors evenly
        # Uncomment this code if you want to visualize misclassifications
        # if incorrect > 0:
        #     indices = [i for i in range(num_classes) if i != seq_idx]
        #     for idx in indices:
        #         cm[seq_idx, idx] = incorrect / (num_classes - 1)

    # Create a plot with a size that scales with the number of classes
    plt.figure(figsize=(min(20, num_classes * 0.3 + 5), min(18, num_classes * 0.3 + 4)))

    # Create a heatmap for the confusion matrix
    # Using original class labels for the axis
    sns.heatmap(cm, cmap="Blues", annot=True, fmt='.0f',
                xticklabels=[idx_to_class[i] for i in range(num_classes)],
                yticklabels=[idx_to_class[i] for i in range(num_classes)])

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
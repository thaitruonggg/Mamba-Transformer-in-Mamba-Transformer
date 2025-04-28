import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Parse the data
data = {
    "Class": [],
    "Accuracy": [],
    "Correct": [],
    "Total": []
}

class_data = """
Class 0 (pl80): 99.28% (276/278)
Class 2 (p6): 97.44% (38/39)
Class 12 (p5): 100.00% (125/125)
Class 13 (pm55): 100.00% (39/39)
Class 14 (pl60): 99.29% (279/281)
Class 15 (ip): 97.79% (133/136)
Class 16 (p11): 99.04% (515/520)
Class 19 (i2r): 97.01% (130/134)
Class 22 (p23): 100.00% (110/110)
Class 39 (pg): 100.00% (46/46)
Class 42 (il80): 100.00% (97/97)
Class 50 (ph4): 97.30% (36/37)
Class 52 (i4): 100.00% (237/237)
Class 55 (pl70): 100.00% (45/45)
Class 61 (pne): 99.70% (671/673)
Class 64 (ph4.5): 95.24% (60/63)
Class 65 (p12): 98.55% (68/69)
Class 66 (p3): 95.08% (58/61)
Class 68 (pl5): 98.08% (204/208)
Class 69 (w13): 100.00% (31/31)
Class 72 (i4l): 99.00% (99/100)
Class 85 (pl30): 99.53% (211/212)
Class 110 (p10): 96.94% (95/98)
Class 111 (pn): 99.90% (1006/1007)
Class 118 (w55): 100.00% (63/63)
Class 128 (p26): 98.47% (258/262)
Class 134 (p13): 91.82% (101/110)
Class 135 (pr40): 100.00% (63/63)
Class 138 (pl20): 100.00% (57/57)
Class 145 (pm30): 96.88% (31/32)
Class 148 (pl40): 98.90% (450/455)
Class 158 (i2): 98.54% (135/137)
Class 162 (pl120): 97.70% (85/87)
Class 168 (w32): 100.00% (37/37)
Class 170 (ph5): 97.67% (42/43)
Class 175 (il60): 97.87% (138/141)
Class 176 (w57): 99.20% (124/125)
Class 179 (pl100): 99.53% (214/215)
Class 183 (w59): 98.41% (62/63)
Class 188 (il100): 100.00% (39/39)
Class 193 (p19): 100.00% (34/34)
Class 216 (pm20): 93.88% (46/49)
Class 220 (i5): 99.42% (512/515)
Class 223 (p27): 100.00% (47/47)
Class 224 (pl50): 99.72% (354/355)
"""

# Process each line
for line in class_data.strip().split('\n'):
    if line:
        # Extract class number and code
        class_info = line.split(':')[0].strip()
        class_num = int(class_info.split()[1])
        class_code = class_info.split('(')[1].split(')')[0]

        # Extract accuracy and counts
        accuracy_part = line.split(':')[1].strip()
        accuracy = float(accuracy_part.split('%')[0])

        # Extract correct and total counts
        counts = accuracy_part.split('(')[1].split(')')[0]
        correct, total = map(int, counts.split('/'))

        # Add to data dictionary
        data["Class"].append(f"{class_num} ({class_code})")
        data["Accuracy"].append(accuracy)
        data["Correct"].append(correct)
        data["Total"].append(total)

# Create a DataFrame
df = pd.DataFrame(data)

# Set up the plot
plt.figure(figsize=(20, 10))

# Create bars
bars = plt.bar(df["Class"], df["Accuracy"], width=1.0, color='skyblue', edgecolor='black')

# Customize the plot
plt.title('Accuracy per Class', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis to start at a reasonable value (e.g., 80%)
plt.ylim(80, 101)

# Add a horizontal line at 95% accuracy for reference
plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% Threshold')

# Highlight bars below 95% accuracy
for i, bar in enumerate(bars):
    if df["Accuracy"][i] < 95:
        bar.set_color('salmon')
        bar.set_edgecolor('black')

# Add text labels for the accuracy values
for i, bar in enumerate(bars):
    accuracy = df["Accuracy"][i]
    correct = df["Correct"][i]
    total = df["Total"][i]
    plt.text(i, accuracy - 0.5, f"{accuracy:.1f}%\n({correct}/{total})",
             ha='center', va='top', rotation=90, fontsize=8)

# Add legend
plt.legend()

# Improve layout
plt.tight_layout()

# Calculate average accuracy
avg_accuracy = sum(df["Correct"]) / sum(df["Total"]) * 100
plt.axhline(y=avg_accuracy, color='green', linestyle='-', alpha=0.5,
            label=f'Average: {avg_accuracy:.2f}%')

# Update legend
plt.legend()

# Show the plot
plt.show()

# Print some statistics
print(f"Average accuracy: {avg_accuracy:.2f}%")
print(f"Number of classes below 95%: {sum(df['Accuracy'] < 95)}")
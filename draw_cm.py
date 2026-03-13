import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix_from_text(text_data, save_path="confusion_matrix.png"):
    """
    Parses a string representation of a confusion matrix and plots a heatmap.
    """
    lines = text_data.strip().split('\n')

    matrix_data = []
    labels = []

    # Skip the first two lines (Header and separator "----")
    for line in lines[2:]:
        if '|' in line:
            # Split the row label from the numeric data
            label, values = line.split('|')
            labels.append(label.strip())

            # Extract numbers
            row_vals = [int(v) for v in values.strip().split()]
            matrix_data.append(row_vals)

    # Convert to a 2D numpy array
    cm = np.array(matrix_data)

    # Plotting
    plt.figure(figsize=(20, 20))

    # annot=True turns on the numbers inside the blocks!
    # fmt='g' keeps them as plain numbers, and annot_kws shrinks the font so they fit.
    sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 10}, cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.title('Confusion Matrix MaMa-MoEx', fontsize=16)

    # Adjust layout so labels don't get cut off
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved successfully as {save_path}")


# --- YOUR NEW DATA ---
raw_text = """Pred ->  Spe   Spe   Spe   Spe   Spe   Spe   End   Spe   Spe   No    No    Rig   Pri   Yie   Sto   No    Veh   No    Gen   Dan   Dan   Dou   Bum   Sli   Roa   Roa   Tra   Ped   Chi   Bic   Bew   Wil   End   Tur   Tur   Ahe   Go    Go    Kee   Kee   Rou   End   End
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Speed |   60     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    0   720     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    0     1   749     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    0     0     0   442     0     7     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    0     0     0     0   659     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    0     0     1     0     0   629     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
End o |    0     0     0     0     0     0   149     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0
Speed |    0     0     0     0     0     0     0   450     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Speed |    1     0     0     1     0    19     0     1   422     0     0     0     0     0     0     6     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
No pa |    0     0     0     0     0     0     0     0     0   480     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
No pa |    0     0     0     0     0     0     0     0     0     0   660     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Right |    0     0     0     0     0     0     0     0     0     0     0   420     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Prior |    0     0     0     0     0     0     0     0     0     0     0     0   685     1     0     4     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Yield |    0     0     0     0     0     0     0     0     0     0     0     0     0   718     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     1     0     0
Stop  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0   270     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
No ve |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   210     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Vehic |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   150     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
No en |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   360     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Gener |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   376     0     1     0     0     0     0     0     1     0     0     0     0    12     0     0     0     0     0     0     0     0     0     0     0
Dange |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    60     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Dange |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    90     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Doubl |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    90     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Bumpy |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   120     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Slipp |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   150     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Road  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0    89     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Road  |    0     0     0     0     0     0     0     0     0     0     0     0     5     0     0     0     0     0     0     0     0     0     0     0     0   475     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Traff |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   180     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Pedes |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    60     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Child |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   150     0     0     0     0     0     0     0     0     0     0     0     0     0     0
Bicyc |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    90     0     0     0     0     0     0     0     0     0     0     0     0     0
Bewar |    0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   149     0     0     0     0     0     0     0     0     0     0     0     0
Wild  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     1     0     0     0     0     0     0     0     0     0     0   269     0     0     0     0     0     0     0     0     0     0     0
End o |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    60     0     0     0     0     0     0     0     0     0     0
Turn  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   210     0     0     0     0     0     0     0     0     0
Turn  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   120     0     0     0     0     0     0     0     0
Ahead |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   390     0     0     0     0     0     0     0
Go st |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0   120     0     0     0     0     0     0
Go st |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    60     0     0     0     0     0
Keep  |    0     0     1     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     2     0     0     0   687     0     0     0
Keep  |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    90     0     0
Round |    0     0     0     0     0     0     0     8     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     2     0     0    80     0     0
End o |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    60     0
End o |    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0    90
"""

plot_confusion_matrix_from_text(raw_text)
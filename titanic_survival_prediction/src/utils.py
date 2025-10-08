# titanic_survival_prediction/src/utils.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

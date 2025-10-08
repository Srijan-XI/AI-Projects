#handwritten_digit_recognition/src/visualize_results.py
import matplotlib.pyplot as plt

def visualize_predictions(images, predictions, actual_labels):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap=plt.cm.gray_r)
        ax.set_title(f"Pred: {predictions[i]}\nActual: {actual_labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

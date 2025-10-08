#iris_flower_classification/src/data_visualization.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairwise(df):
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    plt.show()

def plot_correlation(df):
    corr = df.drop(columns=['species']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

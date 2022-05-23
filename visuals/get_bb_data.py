import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from config import TRAIN_CSV

from config import ROOT_DIR

def scatter_plot(csv_file):
    sns.set_theme()
    warnings.filterwarnings("ignore")
    df = pd.read_csv(csv_file)

    plot = sns.scatterplot(data=df, x="width", y="height", color="#b06048")
    plt.xlabel("Bounding box width [px]")
    plt.ylabel("Bounding box height [px]")
    plt.show()


def dont_ask():
    sns.set(font_scale=1.7)
    ar = np.array([[0.52, 0, 0.1, 0, 0, 0.03, 0.17],
                  [0, 0.21, 0, 0, 0, 0, 0.02],
                  [0, 0.59, 0.10, 0.47, 0, 0.02, 0.26],
                  [0, 0, 0.01, 0.08, 0, 0, 0.03],
                  [0, 0, 0.0, 0, 0.49, 0.01, 0.08],
                  [0.09, 0, 0.11, 0.06, 0, 0.25, 0.44],
                  [0.38, 0.21, 0.76, 0.37, 0.51, 0.70, 0]
                   ])
    y_label = ["IPH", "EDH", "SDH", "Chronic", "IVH", "SAH", "Background FN"]
    x_label = ["IPH", "EDH", "SDH", "Chronic", "IVH", "SAH", "Background FP"]

    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sns.heatmap(ar,
                   annot=True,
                   annot_kws={
                       "size": 20},
                   cmap='Blues',
                   fmt='.2f',
                   square=True,
                   vmin=0.0,
                   xticklabels=x_label,
                   yticklabels=y_label
                    )
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    fig.savefig(os.path.join(ROOT_DIR, "outputs/confusion_matrix.png"), dpi=250)
    plt.close()


if __name__ == "__main__":
    scatter_plot(TRAIN_CSV)
    #dont_ask()

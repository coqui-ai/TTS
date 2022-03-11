import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap

matplotlib.use("Agg")


colormap = (
    np.array(
        [
            [76, 255, 0],
            [0, 127, 70],
            [255, 0, 0],
            [255, 217, 38],
            [0, 135, 255],
            [165, 0, 165],
            [255, 167, 255],
            [0, 255, 255],
            [255, 96, 38],
            [142, 76, 0],
            [33, 0, 127],
            [0, 0, 0],
            [183, 183, 183],
        ],
        dtype=np.float,
    )
    / 255
)


def plot_embeddings(embeddings, num_classes_in_batch):
    num_utter_per_class = embeddings.shape[0] // num_classes_in_batch

    # if necessary get just the first 10 classes
    if num_classes_in_batch > 10:
        num_classes_in_batch = 10
        embeddings = embeddings[: num_classes_in_batch * num_utter_per_class]

    model = umap.UMAP()
    projection = model.fit_transform(embeddings)
    ground_truth = np.repeat(np.arange(num_classes_in_batch), num_utter_per_class)
    colors = [colormap[i] for i in ground_truth]
    fig, ax = plt.subplots(figsize=(16, 10))
    _ = ax.scatter(projection[:, 0], projection[:, 1], c=colors)
    plt.gca().set_aspect("equal", "datalim")
    plt.title("UMAP projection")
    plt.tight_layout()
    plt.savefig("umap")
    return fig

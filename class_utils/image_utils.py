import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def make_montage(image_array, num_cols):
    """
    Returns a montage where the images from the image_array are placed
    next to each other on a grid.

    Args:
        image_array: A 4D numpy image array (batch, width, height, channels);
        num_cols: Number of columns in the montage
    """
    num_rows = int(np.ceil(image_array.shape[0] / num_cols))
    pad = num_cols * num_rows - image_array.shape[0]

    image_array = np.pad(
        image_array,
        (
            (0, pad),
            (0, 0),
            (0, 0),
            (0, 0)
        )
    )

    image_array = image_array.reshape(num_rows, num_cols, *image_array.shape[1:])
    a, b, m, n, c = image_array.shape
    montage = image_array.transpose(0, 2, 1, 3, 4).reshape(a * m, b * n, c)

    return montage

def plot_bboxes(img, bboxes, scores=None, figsize=(10, 8)):
    if isinstance(img, np.ndarray):
        ar = img
    else:
        ar = np.array(img.convert(mode="RGB"))

    if scores is None:
        scores = np.ones(len(bboxes))
    else:
        if len(scores) != len(bboxes):
            raise ValueError("Number of scores must match number of bboxes.")
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.imshow(ar)
    plt.axis("off")

    for (left, top, right, bottom), alpha in zip(bboxes, scores):
        box = patches.Rectangle(
            (left, top),
            width=right-left,
            height=bottom-top,
            linewidth=3, edgecolor='r',
            facecolor='none',
            alpha=alpha
        )

        ax.add_patch(box)
    
    return fig

from dataset import HemorrhageDataset
import os
import random
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw, ImageFont

from config import CLASSES, COLOR_MAP, VALID_DIR, VALID_CSV


def sample_show(image_directory, annotation_directory, sample_index):
    dataset = HemorrhageDataset(image_directory, annotation_directory)
    dataset.set_visual()
    img, target_dict = dataset[sample_index]

    # Get bbox data
    label_index = target_dict["labels"]
    bboxes = target_dict["boxes"]

    # Image draw object
    draw = ImageDraw.Draw(img)

    fontsize = 20
    font_path = "/Users/krystof/Desktop/arial.ttf"
    font = ImageFont.truetype(font_path, fontsize)

    for ind, box in enumerate(bboxes):
        hemorrhage_type = CLASSES[label_index[ind]]
        color = COLOR_MAP[hemorrhage_type]

        coordinates = box.numpy()
        x, y = box[0], box[1]

        draw.text(xy=(x, y - 25), font=font, text=hemorrhage_type, align="right", fill=color)
        draw.rectangle(coordinates, None, color, 2)  # Draw bbox

    img.show()


if __name__ == "__main__":
    im_dir = VALID_DIR
    annotations = VALID_CSV
    for i in range(4):
        ind = random.randint(0, 500)
        sample_show(im_dir, annotations, ind)

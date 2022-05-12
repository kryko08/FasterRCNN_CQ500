import os
import random

import matplotlib.pyplot as plt

from transforms.my_transforms import *
import torchvision.transforms.functional as TF
from dataset import HemorrhageDataset

from config import CLASSES, COLOR_MAP, TRAIN_CSV, TRAIN_DIR

from PIL import Image, ImageDraw, ImageFont


def show_transformed_samples(image_directory, annotation_directory, sample_index):
    dataset = HemorrhageDataset(image_directory, annotation_directory)
    for stage in range(6):
        dataset.set_transform(stage)
        img, target_dict = dataset[sample_index]

        # Tensor to Image
        img = TF.to_pil_image(img)

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
    show_transformed_samples(TRAIN_DIR, TRAIN_CSV, random.randint(0, 500))



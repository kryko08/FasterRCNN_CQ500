from dataset import HemorrhageDataset
import os
import random
import matplotlib.pyplot as plt
import torch

from PIL import Image, ImageDraw, ImageFont

from config import CLASSES, COLOR_MAP


def sample_show(image_directory, annotation_directory, sample_index):
    dataset = HemorrhageDataset(image_directory, annotation_directory)
    dataset.set_visual()
    img, target_dict = dataset[sample_index]

    # Get bbox data
    label_index = target_dict["labels"]
    bboxes = target_dict["boxes"]

    # Image draw object
    draw = ImageDraw.Draw(img)
    #font = ImageFont.

    for ind, box in enumerate(bboxes):
        hemorrhage_type = CLASSES[label_index[ind]]
        color = COLOR_MAP[hemorrhage_type]

        coordinates = box.numpy()  # Tensor to numpy
        print(coordinates)
        draw.rectangle(coordinates, None, color, 3)  # Draw bbox
        draw.text((coordinates[0], coordinates[1]), hemorrhage_type)

    img.show()



# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     # Get object
#     sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
#     img, target = test_dataset[sample_idx]
#
#     # Add to figure
#     figure.add_subplot(rows, cols, i)
#
#     # Image draw object
#     draw_image = ImageDraw.Draw(img)
#
#     label_index = target["labels"]
#     bboxes = target["boxes"]
#     # Draw rectangle for every bounding box
#     for ind, box in enumerate(bboxes):
#         hemorrhage_type = CLASSES[label_index[ind]]
#         color = COLOR_MAP[hemorrhage_type]
#         # Tensor to numpy
#         coordinates = box.numpy()
#         draw_image.rectangle(coordinates, None, color, 3)
#
#         plt.title(f"Sample{i}")
#         plt.axis("off")
#         plt.imshow(img)
# plt.show()

if __name__ == "__main__":
    cwd = os.getcwd()
    annotations = os.path.join(cwd, "../data/train.csv")
    im_dir = os.path.join(cwd, "../data/train")
    ind = random.randint(0, 2000)
    sample_show(im_dir, annotations, ind)

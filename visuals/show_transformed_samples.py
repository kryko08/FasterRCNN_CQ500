import os

import matplotlib.pyplot as plt

from transforms.my_transforms import *
from dataset import HemorrhageDataset

from config import CLASSES, COLOR_MAP

from PIL import Image, ImageDraw


cwd = os.getcwd()
annotations = os.path.join(cwd, "./data/test.csv")
im_dir = os.path.join(cwd, "./data/test")

# transforms_list = [MySaltPepper(5000, 7000), MyBlur(), MyJitter(), MyHorizontalFlip(), MyTensor(), MyRotation(60)]
rotation_list = [MyTensor(), MyRotation(25), MyRotation(30), MyRotation(35), MyRotation(40), MyRotation(45)]

figure = plt.figure(figsize=(8, 8))
row, col = 2, 3
for i, transform in enumerate(rotation_list):
    # Get sample
    test_dataset = HemorrhageDataset(im_dir, annotations, transform)
    #sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
    sample_idx = 150
    sample, target = test_dataset[sample_idx]
    # Add to figure
    figure.add_subplot(row, col, i+1)

    # Get PIL image from tensor
    numpy_array = sample.numpy()

    # Move axis due to ToTensor() transform
    numpy_array = np.moveaxis(numpy_array, 0, -1)

    # 0 - 255
    numpy_array *= 255
    numpy_array = numpy_array.astype(np.uint8)

    pil_image = Image.fromarray(numpy_array)

    draw_image = ImageDraw.Draw(pil_image)

    label_index = target["labels"]
    bboxes = target["boxes"]

    # Draw rectangle for every bounding box
    for ind, box in enumerate(bboxes):
        hemorrhage_type = CLASSES[label_index[ind]]
        color = COLOR_MAP[hemorrhage_type]
        # Tensor to numpy
        coordinates = box.numpy()
        draw_image.rectangle(coordinates, None, color, 3)

        plt.title(transform)
        plt.axis("off")
        plt.imshow(pil_image)
plt.show()





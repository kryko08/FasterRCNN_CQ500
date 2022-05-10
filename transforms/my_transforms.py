import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

import numpy as np

import random

from PIL import Image

import cv2

from transforms.transforms_utils import get_corners, get_enclosing_box, rotate_box, rotate_im

from custom_utils import logger


class MySaltPepper:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound  # around 5000
        self.upper_bound = upper_bound  # around 7000

    @logger
    def __call__(self, image, target):
        """Salt&Pepper"""
        # PIL to numpy nd array
        image_array = np.asarray(image)

        # image size
        row, col, channels = np.shape(image_array)
        # Number of affected pixels
        num_noise = random.randint(self.lower_bound, self.upper_bound)

        # Create "salt" and "pepper"
        for salt_coordinates in range(num_noise // 2):
            # Randomly select x and y coordinate
            x_cor = random.randint(0, row - 1)
            y_cor = random.randint(0, col - 1)
            # Change pixel value across channel dimension
            image_array[x_cor, y_cor, :] = 255

        for pepper_coordinates in range(num_noise // 2):
            # Randomly select x and y coordinate
            x_cor = random.randint(0, row - 1)
            y_cor = random.randint(0, col - 1)
            # Change pixel value across channel dimension
            image_array[x_cor, y_cor, :] = 0

        # Save numpy array to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        tensor = transform(image_array)

        return tensor, target


class MyHorizontalFlip:

    @logger
    def __call__(self, image, target):
        """Flip"""
        # Get bounding box coordinates from target dictionary
        # matrix of N x [x1, y1, x2, y2]
        bbox = target["boxes"]
        bbox = bbox.numpy()

        recalculated = []
        for rectangle in bbox:
            # Flip coordinates

            # y coordinates stay the same
            y1 = rectangle[1]
            y2 = rectangle[3]

            im_width = 512
            width = rectangle[2] - rectangle[0]

            x_center = rectangle[0] + width/2

            # Flip x center
            flipped_x_center = abs(x_center-im_width)
            flipped_x1 = flipped_x_center - width/2
            flipped_x2 = flipped_x1 + width

            recalculated.append([flipped_x1, y1, flipped_x2, y2])

        recalculated = torch.as_tensor(recalculated, dtype=torch.float32)

        target["boxes"] = recalculated

        # PIL image to tensor, rescale to 0 – 1
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        tensor = transform(image)
        # Flip image
        flipped_tensor = TF.hflip(tensor)

        return flipped_tensor, target


class MyJitter:

    @logger
    def __call__(self, image, target):
        """Jitter"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=.5, hue=.3)
        ])

        jitter_tensor = transform(image)

        return jitter_tensor, target


class MyBlur:

    @logger
    def __call__(self, image, target):
        """Blur"""
        # Apply Gaussian noise
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(9, 5))
        ])

        blur_image = transform(image)

        return blur_image, target


class MyRotation:

    def __init__(self, angle):
        self.angle = random.randint(-angle, angle)

    @logger
    def __call__(self, image, target):
        """Rotation"""

        # PIL image to numpy nd array
        image_array = np.array(image)

        # original image parameters
        w, h = 512, 512
        cx, cy = w // 2, h // 2

        rotated_image = rotate_im(image_array, self.angle)

        scale_factor_x = rotated_image.shape[1] / w
        scale_factor_y = rotated_image.shape[0] / h

        rotated_image = cv2.resize(rotated_image, (w, h))

        # To PIL
        pil_image = Image.fromarray(rotated_image)

        # To tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        tensor = transform(pil_image)

        # Get coordinates from "target"
        bbox = target["boxes"]
        bbox = bbox.numpy()

        # Get corner coordinates
        corners = get_corners(bbox)

        corners = np.hstack((corners, bbox[:, 4:]))

        corners[:, :8] = rotate_box(corners[:, :8], self.angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)

        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        # To tensor
        tensor_box = torch.as_tensor(new_bbox, dtype=torch.float32)
        # To dict
        target["boxes"] = tensor_box

        # Recalculate area
        area = (tensor_box[:, 3] - tensor_box[:, 1]) * (tensor_box[:, 2] - tensor_box[:, 0])
        target["area"] = area

        return tensor, target


class MyTensor:

    @logger
    def __call__(self, image, target):
        """Tensor"""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        tensor = transform(image)

        return tensor, target


if __name__ == "__main__":
    from dataset import HemorrhageDataset
    from config import TRAIN_CSV, TRAIN_DIR
    import os
    from torch.utils.data import DataLoader
    from custom_utils import collate_fn

    my_dataset = HemorrhageDataset(os.path.join(os.path.dirname(os.getcwd()), TRAIN_DIR),
                                   os.path.join(os.path.dirname(os.getcwd()), TRAIN_CSV))
    dataloader = DataLoader(my_dataset, batch_size=8, collate_fn=collate_fn)
    it = iter(dataloader)
    for epoch in range(12):
        stage = epoch % 6
        print(stage)
        dataloader.dataset.set_transform(stage)
        batch = next(it)



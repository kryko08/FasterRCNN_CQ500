import os
import pandas as pd

from torch.utils.data import Dataset
from config import CLASSES
from transforms.my_transforms import *


class HemorrhageDataset(Dataset):

    def __init__(self, root, annotations):
        self.root = root
        self.df = pd.read_csv(annotations)

        # Get all images
        self.images = os.listdir(root)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images[idx])
        # Read image
        image = Image.open(image_path).convert("RGB")

        # SOPInstanceUID
        head, tail = os.path.split(image_path)
        sop, ext = os.path.splitext(tail)

        # Get annotations from csv path
        sop_df = self.df[self.df["SOPInstanceUID"] == sop]

        boxes = []
        labels = []
        for index, row in sop_df.iterrows():
            # Get label
            label_name = row["label"]
            label = CLASSES.index(label_name)
            labels.append(label)

            # Get boxes [x1, y1, x2, y2]: upper left corner and lower right corner
            x1 = row["x"]
            y1 = row["y"]

            x2 = row["x"] + row["width"]
            y2 = row["y"] + row["height"]

            boxes.append([x1, y1, x2, y2])

        # Bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # idx to tensor
        image_id = torch.tensor([idx])

        # Final target dictionary
        target = {}
        target["boxes"] = boxes
        target["area"] = area
        target["labels"] = labels
        target["iscrowd"] = is_crowd
        target["image_id"] = image_id

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def set_transform(self, num):
        if num == 0:
            self.transform = MyTensor()
        elif num == 1:
            self.transform = MyJitter()
        elif num == 2:
            self.transform = MyBlur()
        elif num == 3:
            self.transform = MySaltPepper(5000, 7000)
        elif num == 4:
            self.transform = MyRotation(50)
        elif num == 5:
            self.transform = MyHorizontalFlip()

    def set_visual(self):
        self.transform = None

    def __len__(self):
        return len(self.images)

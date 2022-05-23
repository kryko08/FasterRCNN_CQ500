import os

import torch

BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = ["__Background__", "Intraparenchymal", "Epidural", "Subdural", "Chronic", "Intraventricular", "Subarachnoid"]

NUM_CLASSES = len(CLASSES)

COLOR_MAP = {
    "Intraparenchymal": (41, 51, 230),
    "Chronic": (34, 205, 214),
    "Intraventricular": (26, 150, 53),
    "Subdural": (16, 181, 7),
    "Epidural": (224, 74, 204),
    "Subarachnoid": (232, 210, 12)
}


OUT_DIR = "outputs"

TRAIN_DIR = "/Users/krystof/Desktop/split_it/faster/train"
TRAIN_CSV = "/Users/krystof/Desktop/split_it/faster/train.csv"

TEST_DIR = "/Users/krystof/Desktop/split_it/faster/test"
TEST_CSV = "/Users/krystof/Desktop/split_it/faster/test.csv"

VALID_DIR = "/Users/krystof/Desktop/split_it/faster/valid"
VALID_CSV = "/Users/krystof/Desktop/split_it/faster/valid.csv"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch

BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_WORKERS = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = ["__Background__", "Intraparenchymal", "Epidural", "Subdural", "Chronic", "Intraventricular", "Subarachnoid"]

NUM_CLASSES = len(CLASSES)

COLOR_MAP = {
    "Intraparenchymal": "blue",
    "Chronic": "cyan",
    "Intraventricular": "green",
    "Subdural": "lime",
    "Epidural": "pink",
    "Subarachnoid": "yellow"
}

OUT_DIR = "outputs"

TRAIN_DIR = "data/train"
TRAIN_CSV = "data/train.csv"

TEST_DIR = "data/test"
TEST_CSV = "data/test.csv"



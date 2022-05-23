import argparse
import torch
from model import create_model
from config import NUM_CLASSES
import cProfile
import random


def with_grad(mdl):
    mdl.eval()
    inp = [torch.rand(3, 512, 512), torch.rand(3, 512, 512)]
    out = mdl(inp)
    return out


@torch.no_grad()
def no_grad(mdl):
    mdl.eval()
    inp = [torch.rand(3, 512, 512), torch.rand(3, 512, 512)]
    out = mdl(inp)
    return out


if __name__ == "__main__":
    model = create_model(NUM_CLASSES)
    cProfile.run("with_grad(model)")

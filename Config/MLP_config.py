import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.MLP import MLP
from Models.MLP import training

POSITIVE_RATIO = 0.5
train, test, sample, dataset = make_dataset(700, 500, 2500, POSITIVE_RATIO, 'synthetic')

model = MLP(train.x.shape[1], 1)
loss_list = []
training(model, train, 50)

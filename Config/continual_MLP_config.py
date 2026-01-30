import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Dataset.makeDataset import make_dataset
from Models.MLP import MLP
from Models.synapticIntelligence import SynapticIntelligence, continual_training

POSITIVE_RATIO = 0.5
train, test, sample, dataset = make_dataset(700, 500, 2500, POSITIVE_RATIO, 'synthetic')
model = MLP(train.x.shape[1], 1)
loss_list = []
si = SynapticIntelligence(model)

continual_training(si, train, 100, loss_list, tao=0)
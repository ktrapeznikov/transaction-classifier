from models import TransClassifier, read_and_process_data, train_tokenizer, compute_features, prepair_training_dataset, \
    inspect, device
import pytorch_lightning as pl
from argparse import Namespace
from logging import getLogger
import torch
import os
from shutil import copyfile

logger = getLogger(__name__)


def evaluate():
    data = read_and_process_data("transactions_training_data.csv", after="2020-01-01")
    features_ids = compute_features(data)
    dataset = prepair_training_dataset(features_ids, data)
    inspect("model.ckpt", dataset, data)


if __name__ == "__main__":
    evaluate()

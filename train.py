from models import TransClassifier, read_and_process_data, train_tokenizer, compute_features, prepair_training_dataset, \
    inspect, device
import pytorch_lightning as pl
from argparse import Namespace
from logging import getLogger
import torch
import os
from shutil import copyfile

logger = getLogger(__name__)



def train():
    hparams = Namespace(gpus=1 if device == "cude" else None,
                        dropout_rate=.2,
                        hidden_dim=32,
                        batch_size=256,
                        seq_type="cnn",
                        max_epochs=100,
                        min_epochs=10,
                        progress_bar_refresh_rate=1,
                        best_model_path="model.ckpt")

    data = read_and_process_data("transactions_training_data.csv", after='2017-07-01', before="2019-12-31")
    train_tokenizer(data)
    features_ids = compute_features(data)
    dataset = prepair_training_dataset(features_ids, data)

    model = TransClassifier(hparams)
    trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                         min_epochs=hparams.min_epochs,
                         gpus=hparams.gpus,
                         progress_bar_refresh_rate=hparams.progress_bar_refresh_rate)

    trainer.fit(model)

    copyfile(trainer.checkpoint_callback.kth_best_model, hparams.best_model_path)

    inspect(hparams.best_model_path, dataset, data)


if __name__ == "__main__":
    train()

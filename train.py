from models import TransClassifier, read_and_process_data, train_tokenizer, compute_features, prepair_training_dataset
import pytorch_lightning as pl
from argparse import Namespace
from logging import getLogger
import torch
import os
from shutil import copyfile

logger = getLogger(__name__)


def train():
    hparams = Namespace(dataset="training_dataset.pkl",
                        gpus=None,
                        dropout_rate=.2,
                        hidden_dim=32,
                        batch_size=256,
                        seq_type="cnn",
                        max_epochs=100,
                        min_epochs=10,
                        progress_bar_refresh_rate=1,
                        best_model_path="model.ckpt")

    device = "cuda" if hparams.gpus is not None else "cpu"

    data = read_and_process_data("transactions_training_data.csv", after='2017-07-01')
    train_tokenizer(data)
    features_ids = compute_features(data)
    dataset = prepair_training_dataset(features_ids, data, save_file=hparams.dataset)

    model = TransClassifier(hparams)
    trainer = pl.Trainer(max_epochs=hparams.max_epochs,
                         min_epochs=hparams.min_epochs,
                         gpus=hparams.gpus,
                         progress_bar_refresh_rate=hparams.progress_bar_refresh_rate)

    trainer.fit(model)

    copyfile(trainer.checkpoint_callback.kth_best_model, hparams.best_model_path)

    evaluate(hparams.best_model_path, dataset, data)


def evaluate(model_path, dataset, data):
    model = TransClassifier.load_from_checkpoint(model_path)
    # check how model did
    with torch.no_grad():
        x_s, x_f, y, w = dataset.tensors
        out = model(x_s.to(device), x_f.to(device))
        probs = out[0].softmax(dim=1)[:, 1].cpu().numpy()
    wrong = ((probs > .5) != data.label.values)
    logger.info(f"mean error {wrong.mean()}")
    data["probs"] = probs
    data["wrong"] = wrong
    logger.info("top false positives")
    s = data.query("wrong").sort_values("probs").tail(20).loc[:,
        ["Date", "Original Description", "Labels", "Amount", "probs"]]
    logger.info(f"\n{s}")
    logger.info("top false negatives")
    s = data.query("wrong").sort_values("probs").head(20).loc[:,
        ["Date", "Original Description", "Labels", "Amount", "probs"]]
    logger.info(f"\n{s}")


if __name__ == "__main__":
    train()

import pickle
from argparse import Namespace
import pandas as pd
import os
import numpy as np
import argparse
import youtokentome as yttm
import torch
from torch.nn import functional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset
from collections import defaultdict
import logging

logger = logging.getLogger()


# def set_up_logger():
#     # Configure logging
#     log_format = '%(levelname)-10s %(asctime)s %(filename)-35s %(funcName)-15s %(lineno)-5d: %(message)s'
#     formatter = logging.Formatter(log_format)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#     logger.setLevel(logging.INFO)


# set_up_logger()


def read_and_process_data(file_name, before='2030-01-01', after='2000-01-01'):
    data = pd.read_csv("transactions_training_data.csv", parse_dates=["Date"]).query(
        f"Date>='{after}' & Date<='{before}'")
    sep = " "
    data["feature_string"] = (data.Category + sep +
                              data.Date.dt.day_name() + sep +
                              data["Account Name"] + sep +
                              data["Transaction Type"] + sep +
                              data["Original Description"]).str.lower()

    data["label"] = ~data.Labels.isna()
    data["feature_float"] = data.Amount
    data["weights"] = np.ones_like(data.Amount)
    # data["weights"] = data.Amount/ data.Amount.mean()

    logger.info(data["label"].mean())

    return data


def train_tokenizer(data, model_path="vocab.model", vocab_size=1000):
    train_data_path = "train_data.txt"
    open(train_data_path, "w").write("\n".join(data.feature_string.values))
    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_path)


def compute_features(data, model_path="vocab.model", max_len=20):
    bpe = yttm.BPE(model=model_path)
    features_ids = bpe.encode(data.feature_string.values.tolist(), output_type=yttm.OutputType.ID)
    features_ids = [f[:max_len] + [0] * (max_len - len(f)) for f in features_ids]
    return np.array(features_ids)


def prepair_training_dataset(features_ids, data, save_file="training_dataset.pkl"):
    x_s = features_ids
    x_f = data["Amount"].values.reshape(-1, 1)
    y = data["label"].values
    w = data["weights"].values

    dataset = TensorDataset(torch.tensor(x_s, dtype=torch.long),
                            torch.tensor(x_f, dtype=torch.float),
                            torch.tensor(y, dtype=torch.long),
                            torch.tensor(w, dtype=torch.float))

    if save_file is not None:
        pickle.dump(dataset, open(save_file, "wb"))

    return dataset



def acc(y, logits):
    yhat = logits.argmax(dim=1)
    return (1.0 * (yhat == y)).mean()


class TransClassifier(pl.LightningModule):
    default_hparams = {"hidden_dim": 32,
                       "batch_size": 256,
                       "dropout_rate": .2,
                       "num_emb": 1000,
                       "seq_type": "cnn",
                       "cont_dim": 1,
                       "kernel_size": 3,
                       "dataset": None}

    def __init__(self, hparams: Namespace = None):

        super().__init__()

        hparams = self.check_hparams(hparams)
        self.hparams = hparams

        if hparams.dataset is None:
            raise ValueError(f"must specify dataset for training")

        padding_idx = 0 if hparams.seq_type is None else None

        self.emb = torch.nn.Embedding(hparams.num_emb, embedding_dim=hparams.hidden_dim, padding_idx=padding_idx)

        if hparams.seq_type == "cnn":
            self.seq_encoder = torch.nn.Conv1d(hparams.hidden_dim, hparams.hidden_dim, hparams.kernel_size)
        elif hparams.seq_type == "lstm":
            self.seq_encoder = torch.nn.LSTM(hparams.hidden_dim, hparams.hidden_dim, batch_first=True)
        else:
            hparams.seq_encoder = None

        self.cont_lin = torch.nn.Linear(hparams.cont_dim, hparams.cont_dim)
        self.cls = torch.nn.Linear(hparams.hidden_dim+hparams.cont_dim, 2)
        self.drop = torch.nn.Dropout(hparams.dropout_rate)

        self.dataset = None
        self.dataset_val = None
        self.dataset_train = None

    def check_hparams(self, hparams):
        temp = self.default_hparams.copy()
        hparams_dict = vars(hparams)
        for k, v in hparams_dict.items():
            if k not in temp:
                logger.warning(f"unknown {k} in hparams")

        temp.update(hparams_dict)
        return Namespace(**temp)

    def input_drop_out(self, x):
        if self.training:
            to_zero = torch.rand(*x.shape) <= self.hparams.dropout_rate
            x[to_zero] = 0
        return x

    def forward(self, x_s, x_f, y=None, w=None):

        x_s = self.input_drop_out(x_s)
        x_emb = self.emb(x_s)

        if self.hparams.seq_type == "cnn":
            x = self.seq_encoder(x_emb.permute(0, 2, 1))  # .permute(0,1,2)
            x_reduced, _ = x.max(dim=2)

        elif self.hparams.seq_type == "lstm":
            x = self.seq_encoder(x_emb)
            x_reduced = x[0].mean(dim=1)
        else:
            x_reduced = x_emb.mean(dim=1)

        # x_comb = x_reduced + self.cont_lin(x_f)
        x_comb = torch.cat([x_reduced, self.cont_lin(x_f)], dim=1)
        x_comb = self.drop(x_comb)
        x_comb = torch.nn.functional.relu(x_comb)
        logits = self.cls(x_comb)

        out = (logits,)

        if y is not None:
            temp = functional.cross_entropy(logits, y, reduction='none')
            if w is not None:
                loss = (temp * w).mean()
            else:
                loss = temp.mean()

            out = out + (loss, acc(y, logits))

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _step(self, batch, batch_idx):
        x_s, x_f, y, w = batch
        yhat, l, a = self.forward(x_s, x_f, y, w)
        return dict(loss=l, acc=a)

    def training_step(self, batch, batch_idx):
        out_dict = self._step(batch, batch_idx)
        return dict(loss=out_dict["loss"], log=out_dict)

    def validation_step(self, batch, batch_idx):
        out_dict = self._step(batch, batch_idx)
        return {f'val_{k}': v for k, v in out_dict.items()}

    def validation_epoch_end(self, outputs):
        temp = defaultdict(list)
        for output in outputs:
            for k, v in output.items():
                temp[k].append(v)

        for k, v in temp.items():
            temp[k] = torch.stack(v).mean()

        temp = {k: v for k, v in temp.items()}

        temp["log"] = {k: v for k, v in temp.items()}
        return temp

    def prepare_data(self):

        self.dataset = pickle.load(open(self.hparams.dataset, "rb"))

        ntrain = int(.75 * len(self.dataset))
        dataset_train, dataset_val = random_split(self.dataset, [ntrain, len(self.dataset) - ntrain])

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, shuffle=False)



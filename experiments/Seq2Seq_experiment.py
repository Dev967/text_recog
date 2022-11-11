import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CONF import batch_size, shuffle
from datasets.IAM_words import train_dataset, test_dataset
from models.Seq2Seq import Encoder, Decoder
from runs.Seq2SeqRun import Run
from utils.data.custom_collate_fn import collate_variable_images
from utils.run_conf import RunConf


class Experiment:
    def __init__(self):
        # self.trained_model = 'outputs/logged_models/02_11_2022/18_40/network'

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                           collate_fn=collate_variable_images)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                          collate_fn=collate_variable_images)

        config = {
            "name": "Seq2Seq GRU",
            "description": "Seq2Seq model using GRU ",
            "model_path": None,
            "quick_run": False,  # Set true if dont want to save the stats or train the model
            "train": True,  # Set false if dont need to train the model
            "show_plots": True,

            "epochs": 5,
            "lr": 0.1,

            "loss_fn": nn.NLLLoss(),
            "optim_fn": torch.optim.SGD,

            "batch_size": 64,
            "train_dataloader": self.train_dataloader,
            "test_dataloader": self.test_dataloader,
        }

        self.run_conf = RunConf(config)

    def start(self):
        run_handle = Run(Encoder, Decoder, self.run_conf)
        run_handle.run()

    def test(self):
        run_handle = Run(Encoder, Decoder, self.run_conf)
        # run_handle.test()
        # run_handle.performance_metrics(True )
        # run_handle.compare_lr([0.0001, 0.001, 0.01, 0.1, 0.9], self.test_dataloader, True)
        # run_handle.print_conv_filters()

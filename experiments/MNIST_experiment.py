import torch
import torch.nn as nn
from datasets.MNIST import train_ds, test_ds
from runs.ConvBasicRun import Run
from models.BasicConv import Network
from utils.run_conf import RunConf


class Experiment:
    def __init__(self):
        # self.trained_model = 'outputs/logged_models/02_11_2022/18_40/network'

        config = {
            "name": "MNIST Conv Basic",
            "description": "Basic Convolutional MNIST classification using Convolutions",
            "model_path": None,
            "quick_run": False,  # Set true if dont want to save the stats or train the model
            "train": True,  # Set false if dont need to train the model
            "show_plots": True,

            "epochs": 5,
            "lr": 0.1,

            "loss_fn": nn.CrossEntropyLoss(),
            "optim_fn": torch.optim.SGD,

            "batch_size": 64,
            "train_ds": train_ds,
            "test_ds": test_ds,
        }

        self.run_conf = RunConf(config)

    def start(self):
        run_handle = Run(Network, self.run_conf)
        run_handle.eda(train_ds, False)
        run_handle.compare_lr([0.0001, 0.001, 0.01, 0.1, 0.9], None, False)
        run_handle.run()
        run_handle.performance_metrics(False)

    def test(self):
        run_handle = Run(Network, self.run_conf)
        run_handle.test()
        # run_handle.performance_metrics(True )
        # run_handle.compare_lr([0.0001, 0.001, 0.01, 0.1, 0.9], None, True)
        # run_handle.print_conv_filters()

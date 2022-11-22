import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CONF import batch_size, shuffle
from datasets.IAM_words import train_dataset, test_dataset, lang
from models.Seq2Seq import Encoder, AttentionDecoder
from runs.Seq2SeqRun import Run
from utils.data.custom_collate_fn import collate_variable_images
from utils.run_conf import RunConf
from utils.training_script.segmented_model_train import TrainingType


class Experiment:
    def __init__(self):
        # self.trained_model = 'outputs/logged_models/02_11_2022/18_40/network'

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                           collate_fn=collate_variable_images)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                          collate_fn=collate_variable_images)

        config = {
            "name": "Seq2Seq LSTM with Attention",
            "description": "More LSTM layers, Tanh, using LSTM instead of GRU",
            "model_path": None,
            "quick_run": False,  # Set true if you don't want to save the stats or train the model
            "train": True,  # Set false if you don't need to train the model
            "show_plots": True,

            "epochs": 10,
            "lr": 0.01,

            "loss_fn": nn.NLLLoss(),
            "optim_fn": torch.optim.SGD,

            "batch_size": 64,
            "train_dataloader": self.train_dataloader,
            "test_dataloader": self.test_dataloader,
            "train_type": TrainingType.ATTENTION,
            "encoder": Encoder(bidirectional=False, num_layers=1),
            "decoder": AttentionDecoder(80, 80, 128, bidirectional=False, num_layers=1)
        }

        self.run_conf = RunConf(config)

    def start(self):
        run_handle = Run(run_conf=self.run_conf)
        # run_handle = BasicSeq2SeqRun(Encoder, Decoder, self.run_conf)
        run_handle.run()

    def test(self):
        sample_images, sample_labels = next(iter(self.train_dataloader))

        # run_handle = AttentionSeq2SeqRun(Encoder, AttentionDecoder, self.run_conf)
        # run_handle = BasicSeq2SeqRun(Encoder, Decoder, self.run_conf)

        # run_handle.test()
        # run_handle.performance_metrics(True )
        # run_handle.compare_lr([0.0001, 0.001, 0.01, 0.1, 0.9], True)
        # run_handle.print_conv_filters()

    def evaluate(self):
        sample_images, sample_labels = next(iter(self.train_dataloader))
        # img = sample_images[0]
        # label = sample_labels[0]

        # plt.imshow(img, cmap='gray')
        # plt.show()

        encoder = torch.load('Trained Models/Seq2Seq_Basic/encoder')
        decoder = torch.load('Trained Models/Seq2Seq_Basic/decoder')

        loss = 0
        loss_fn = nn.NLLLoss()
        for idx in range(len(sample_images)):
            img = sample_images[idx]
            label = sample_labels[idx]

            encoder_hidden = encoder.init_hidden()
            encoder_output, encoder_hidden = encoder(img.unsqueeze(1), encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_input = lang.indexEncoding(torch.Tensor([0]).long())
            decoder_outputs = torch.Tensor()

            for i in range(len(label)):
                output, decoder_hidden = decoder(decoder_input.view(1, 1, -1), decoder_hidden)
                decoder_outputs = torch.cat((decoder_outputs, output.squeeze(1)), 0)
                decoder_input = output
                topv, topk = output.topk(1)
                if topk == 1: break

            print(decoder_outputs.argmax(1), label)
            print(">", lang.tensorToWord(label))
            print(f'= {lang.tensorToWord(decoder_outputs.argmax(1))}')

            loss += loss_fn(decoder_outputs, label)

        print("LOSS: ", loss)

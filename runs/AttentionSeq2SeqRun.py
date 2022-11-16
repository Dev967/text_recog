import math
import os
import time
from datetime import datetime

import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, random_split

from CONF import *
from utils.data.custom_collate_fn import collate_variable_images
from utils.training_script.segmented_model_train import train


class Run:
    def __init__(self, Encoder, Decoder, run_conf):
        self.uniq_classes = None
        self.name = run_conf.name
        self.run_conf = run_conf
        self.Encoder = Encoder
        self.Decoder = Decoder

        if run_conf.model_path:
            self.encoder = torch.load(run_conf.model_path[0])
            self.decoder = torch.load(run_conf.model_path[1])
        else:
            self.encoder = Encoder(128, 128)
            self.decoder = Decoder(80, 80, 128)

        self.optim_fn = run_conf.optim_fn([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=run_conf.lr)

        self.train_dataloader = run_conf.train_dataloader
        self.test_dataloader = run_conf.test_dataloader

        self.sample_images, self.sample_labels = next(iter(self.train_dataloader))

        now = datetime.now()
        today = now.strftime("%d_%m_%Y")
        curr_time = now.strftime("%H_%M")
        self.curr_dir = f'outputs/logged_models/{today}/{curr_time}'

    def run(self, local_run_conf=None):

        if local_run_conf: self.run_conf = local_run_conf

        if not self.run_conf.quick_run:
            # logging stuff
            os.makedirs(self.curr_dir, exist_ok=True)
            file_obj = open(f'{self.curr_dir}/description.txt', 'a')
            file_obj.write(
                f'{self.name} \n\n {self.run_conf.description} \n Learning Rate: {self.run_conf.lr} \n Optimizer: {str(self.optim_fn)} \n Loss function: {str(self.run_conf.loss_fn)}\n')

            if self.run_conf.train:
                # training
                tic = time.perf_counter()
                os.makedirs(f'{self.curr_dir}/losses', exist_ok=True)
                for epoch in range(self.run_conf.epochs):
                    print(f'Epoch {epoch}')
                    loss = train(self.encoder, self.decoder, self.train_dataloader, self.run_conf.loss_fn,
                                 self.optim_fn, verbose=True, attention_enabled=True)

                    torch.save(self.encoder, f'{self.curr_dir}/encoder')
                    torch.save(self.decoder, f'{self.curr_dir}/decoder')
                    torch.save(loss, f'{self.curr_dir}/losses/epoch_{epoch}')

                    print(f'\nLoss after epoch({epoch}): ', sum(loss) / len(loss))

                toc = time.perf_counter()
                time_taken = (toc - tic) / 60

                file_obj.write(f'\n Time Taken: {time_taken} min\n')
                print(f"Finished all Epochs in {time_taken} minutes")

            # save model after training
            torch.save(self.encoder, f'{self.curr_dir}/encoder')
            torch.save(self.decoder, '{self.curr_dir}/encoder')

    def compare_lr(self, learning_rates, show_plots):
        total = len(self.train_dataloader.dataset)
        subset_size = math.floor(total / 10)
        rest = total - subset_size
        subset, _ = random_split(self.train_dataloader.dataset, [subset_size, rest])
        subset_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=collate_variable_images)

        print('comparing learning rates')
        os.makedirs(f'{self.curr_dir}/plots', exist_ok=True)
        if self.run_conf.model_path: return

        losses = []
        for lr in learning_rates:
            print(f'Testing {lr}...')
            encoder = self.Encoder(128, 128)
            decoder = self.Decoder(80, 80, 128)
            loss_func = nn.NLLLoss()
            optim_fn = torch.optim.SGD([
                {'params': encoder.parameters()},
                {'params': decoder.parameters()}
            ], lr=lr)
            loss_arr = train(encoder, decoder, subset_dataloader, loss_func, optim_fn, verbose=True,
                             attention_enabled=True)
            losses.append(loss_arr)

        rows, cols, idx = 2, 3, 0
        subs = make_subplots(rows=rows, cols=cols, subplot_titles=learning_rates)

        for row in range(rows):
            for col in range(cols):
                if row == 1 and col == 2: break
                subs.add_trace(go.Scatter(y=losses[idx]), row=row + 1, col=col + 1)
                idx += 1

        subs.write_image(f'{self.curr_dir}/plots/learning_rates.png')
        if show_plots: subs.show()
        print('done\n')

    def test(self):
        print("sample image: ", len(self.sample_images), self.sample_images[0].shape, " Sample labels: ",
              len(self.sample_labels), self.sample_labels[0].shape)

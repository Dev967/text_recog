import torch
from torch.utils.data import DataLoader
from datetime import datetime
import os
import time
from torchinfo import summary
from torchvision.utils import make_grid

from utils.training_script.v1 import train
from utils.testing_script.v1 import test
from utils.data.get_unique_classes import makeUniqueTensor
from utils.metrics.confusion_matrix import build_confusion_matrix
from utils.metrics.learning_rates import test_learning_rates

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl


class Run:
    def __init__(self, Network, run_conf):
        self.uniq_classes = None
        self.name = run_conf.name
        self.run_conf = run_conf
        self.Network = Network

        if run_conf.model_path:
            self.myNet = torch.load(run_conf.model_path)
        else:
            self.myNet = Network()

        self.optim_fn = run_conf.optim_fn(self.myNet.parameters(), lr=run_conf.lr)

        # dataloader
        self.train_dataloader = DataLoader(run_conf.train_ds, batch_size=run_conf.batch_size, shuffle=True)
        if run_conf.test_ds: self.test_dataloader = DataLoader(run_conf.test_ds, batch_size=run_conf.batch_size,
                                                               shuffle=True)
        self.sample_images, self.sample_labels = next(iter(self.train_dataloader))

        now = datetime.now()
        today = now.strftime("%d_%m_%Y")
        curr_time = now.strftime("%H_%M")
        self.curr_dir = f'outputs/logged_models/{today}/{curr_time}'

    def run(self, local_run_conf=None):

        if local_run_conf: self.run_conf = local_run_conf

        if not self.run_conf.quick_run:
            os.makedirs(self.curr_dir, exist_ok=True)
            file_obj = open(f'{self.curr_dir}/description.txt', 'a')
            file_obj.write(
                f'{self.name} \n\n {self.run_conf.description} \n Learning Rate: {self.run_conf.lr} \n Optimizer: {str(self.optim_fn)} \n Loss function: {str(self.run_conf.loss_fn)}\n')

            if self.run_conf.train:
                # training
                tic = time.perf_counter()
                for epoch in range(self.run_conf.epochs):
                    loss = train(self.myNet, self.train_dataloader, self.run_conf.loss_fn, self.optim_fn, verbose=False)
                    # for checkpoint, incase training is interrupted in-between
                    torch.save(self.myNet, f'{self.curr_dir}/network')
                    print("\nAverage Loss ", sum(loss) / len(loss))
                    self.print_conv_filters(epoch=epoch)

                toc = time.perf_counter()
                time_taken = (toc - tic) / 60

                file_obj.write(f'\n Time Taken: {time_taken} min\n')
                print(f"Finished training in {time_taken} minutes")

            # save model after training
            torch.save(self.myNet, f'{self.curr_dir}/network')

            # validation
            [loss, correct_pred, count] = test(self.myNet, self.test_dataloader, self.run_conf.loss_fn)
            avg_loss = loss / count
            accuracy = correct_pred / count
            print(f'\n AVG Loss: {avg_loss} Accuracy: {accuracy}')
            file_obj.write(f'\n Accuracy: {accuracy} \n AVG Loss: {avg_loss} ')

            # summary
            model_stats = summary(self.myNet, (self.run_conf.batch_size, 1, 28, 28))
            file_obj.write(f'\n\n\n {str(model_stats)}')

    def print_conv_filters(self, epoch):
        print("printing filters....")
        # if not self.uniq_classes: self.eda(self.test_dataloader.dataset, False)

        layer_input = self.uniq_classes.unsqueeze(dim=1)
        counter = 0
        for layer in self.myNet.conv_stack:
            if "Conv2d" in str(layer):
                plot_path = f'{self.curr_dir}/plots/{epoch}/Conv2d_{counter}'
                os.makedirs(plot_path, exist_ok=True)
                output = layer(layer_input)
                layer_input = output
                output = output.transpose(1, 0)
                for idx, x in enumerate(output):
                    grid = make_grid(x.unsqueeze(dim=1))
                    plt.imshow(grid.permute(1, 2, 0))
                    plt.savefig(f'{plot_path}/kernel_{idx}.png')
                counter += 1
        print("done \n")

    def eda(self, dataset, show_plots):
        print("processing eda...")
        os.makedirs(f'{self.curr_dir}/plots', exist_ok=True)
        targets = [y for X, y in dataset]
        plt.hist(targets)
        plt.savefig(f'{self.curr_dir}/plots/class_balance.png')
        if show_plots: plt.show()
        plt.clf()

        uniqClasses = makeUniqueTensor(self.sample_images, self.sample_labels)
        self.uniq_classes = uniqClasses
        grid = make_grid(uniqClasses.unsqueeze(dim=1), nrow=5)
        plt.imshow(grid.permute(1, 2, 0))
        plt.savefig(f'{self.curr_dir}/plots/classes.png')
        if show_plots: plt.show()
        plt.clf()
        print("eda finished.\n")

    def performance_metrics(self, show_plots):
        print("calculating performance metrics")
        # if not self.uniq_classes: self.eda(self.test_dataloader.dataset, False)
        conf_matrix = build_confusion_matrix(self.myNet, self.train_dataloader, self.uniq_classes).detach().numpy()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=mpl.colormaps['viridis'], alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig(f'{self.curr_dir}/plots/confusion_matrix.png')
        if show_plots: plt.show()
        plt.clf()
        print('done\n')

    def compare_lr(self, learning_rates, dataloader, show_plots):
        print('comparing learning rates')
        os.makedirs(f'{self.curr_dir}/plots', exist_ok=True)
        if self.run_conf.model_path: return

        if not dataloader: dataloader = self.train_dataloader
        total_losses = test_learning_rates(self.Network, dataloader, learning_rates, train)

        rows, cols, idx = 2, 3, 0
        subs = make_subplots(rows=rows, cols=cols, subplot_titles=learning_rates)

        for row in range(rows):
            for col in range(cols):
                if row == 1 and col == 2: break;
                subs.add_trace(go.Scatter(y=total_losses[idx]), row=row + 1, col=col + 1)
                idx += 1

        subs.write_image(f'{self.curr_dir}/plots/learning_rates.png')
        if show_plots: subs.show()
        print('done\n')

    def test(self):
        print("sample image: ", self.sample_images.shape, " Sample labels: ", self.sample_labels.shape)
        output = self.myNet(self.sample_images)
        print("Output: ", output.shape, output.flatten(start_dim=1).shape)

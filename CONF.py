import socket

import torch.cuda

host = socket.gethostname()

dataset_dir = '/home/dev/winstorage/arch/main/Workspace/PRIMARY/Python/data/IAM' if host == "localhost" else '/kaggle/input/iam-dataset/datasets'

image_dir = f'{dataset_dir}/words'
target_file = f'{dataset_dir}/ascii/words.txt' if host == "localhost" else f'{dataset_dir}/words.txt'

batch_size = 64
shuffle = True
device = "cuda" if torch.cuda.is_available() else "cpu"

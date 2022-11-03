import torch.cuda

dataset_dir = '/home/dev/winstorage/arch/main/Workspace/PRIMARY/Python/data/IAM/words'
target_file = '/home/dev/winstorage/arch/main/Workspace/PRIMARY/Python/data/IAM/ascii/words.csv'
IIT5K_path = '/home/dev/winstorage/arch/main/Workspace/PRIMARY/Python/data/IIIT5K'
MNIST_path = '/home/dev/winstorage/arch/main/Workspace/PRIMARY/Python/data/MNIST/'
batch_size = 64
shuffle = True
device = "cuda" if torch.cuda.is_available() else "cpu"

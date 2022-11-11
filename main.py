from CONF import host
from experiments.Seq2Seq_experiment import Experiment


def main():
    print("Running on ", host)
    exp = Experiment()
    exp.start()


main()

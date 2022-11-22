from CONF import host
from experiments.Attention_Seq2Seq import Experiment as AttentionExperiment


def main():
    print("Running on ", host)
    exp = AttentionExperiment()
    exp.start()
    # exp.test()


main()

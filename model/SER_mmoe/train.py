from SER_mmoe.utils.parser import Arguments
from SER_mmoe.utils.runs import Run
from SER_mmoe.training.training import train


def main():
    parser = Arguments(description='Training with MLM.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    args = parser.parse()

    with Run.context(consider_failed_if_interrupted=False):
        model = train(args)


if __name__ == "__main__":
    main()

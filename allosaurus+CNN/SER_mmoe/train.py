from SER_mmoe.utils.parser import Arguments
from SER_mmoe.training.training import train


def main():
    parser = Arguments(description='Training')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_training_input()

    args = parser.parse()

    train(args)


if __name__ == "__main__":
    main()

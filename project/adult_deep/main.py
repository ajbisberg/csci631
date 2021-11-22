import argparse
from torch.utils.data import DataLoader

from dataset import AdultDataset
from model import AdultModel
from train import train


def main(train_data_path, use_dp):
    train_data = AdultDataset(train_data_path)

    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = AdultModel()

    train(model, train_dataloader, batch_size=128, use_dp=use_dp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description='Preprocess Adult data and train deep model')
    parser.add_argument('--train-data-path',
                        help='Path to Adult training data (CSV)',
                        required=True)
    parser.add_argument('--use-dp',
                        help='Use DP-SGD',
                        action='store_true')
    args = parser.parse_args()

    main(args.train_data_path, args.use_dp)

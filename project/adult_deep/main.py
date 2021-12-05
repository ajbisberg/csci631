import argparse
import pickle
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from model import AdultModel
from train import train


def main(data_path, use_dp, batch_size=128):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_rows = Tensor(data['train_rows'])
    train_labels = Tensor(data['train_labels'])
    train_dataset = TensorDataset(train_rows, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    cross_val_rows = Tensor(data['cross_val_rows'])
    cross_val_labels = Tensor(data['cross_val_labels'])
    cross_val_dataset = TensorDataset(cross_val_rows, cross_val_labels)
    cross_val_dataloader = DataLoader(cross_val_dataset)

    test_rows = Tensor(data['test_rows'])
    test_labels = Tensor(data['test_labels'])
    test_dataset = TensorDataset(test_rows, test_labels)
    test_dataloader = DataLoader(test_dataset)

    model = AdultModel()

    train(model, train_dataloader, cross_val_dataloader, batch_size=batch_size, use_dp=use_dp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description='Train deep model')
    parser.add_argument('--data-path',
                        help='Path to pickled data',
                        required=True)
    parser.add_argument('--use-dp',
                        help='Use DP-SGD',
                        action='store_true')
    args = parser.parse_args()

    main(args.data_path, args.use_dp)

import argparse

from data import DatasetChoice, load_data
from model import AdultModel, HousingModel
from train import DpSgdParameters, train, TrainingParameters


def main(dataset_choice: DatasetChoice, data_path, training_params: TrainingParameters):
    train_dataloader, cross_val_dataloader, test_dataloader = load_data(dataset_choice, data_path, training_params.batch_size)

    if dataset_choice == DatasetChoice.HOUSING:
        model = HousingModel()
    elif dataset_choice == DatasetChoice.ADULT:
        model = AdultModel()
    else:
        raise NotImplementedError('No model available for dataset choice.')

    model = train(model, train_dataloader, cross_val_dataloader, training_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description='Train deep model')
    parser.add_argument('--data-path',
                        help='Path to pickled data',
                        required=True)
    parser.add_argument('--dataset',
                        choices=['HOUSING', 'ADULT'],
                        help='Choice of dataset, adult or housing',
                        required=True)
    parser.add_argument('--num-epochs',
                        help='Number of epochs in training',
                        type=int,
                        default=10)
    parser.add_argument('--batch-size',
                        help='Training batch size for SGD',
                        type=int,
                        default=128)
    parser.add_argument('-lr', '--learning-rate',
                        help='Optimizer learning rate',
                        type=float,
                        default=1e-4)
    parser.add_argument('--use-dp',
                        help='Whether to use SGD or DP-SGD',
                        action='store_true')
    parser.add_argument('--target-epsilon',
                        help='Target epsilon for DP-SGD',
                        type=float,
                        default=1)
    parser.add_argument('--target-delta',
                        help='Target delta for DP-SGD',
                        type=float,
                        default=1e-5)
    parser.add_argument('--max-grad-norm',
                        help='Max gradient norm for DP-SGD',
                        type=float,
                        default=1.0)
    args = parser.parse_args()

    dataset_choice = DatasetChoice[args.dataset]

    if args.use_dp:
        kwargs = {
            'target_epsilon': args.target_epsilon,
            'target_delta': args.target_delta,
        }
        if args.max_grad_norm:
            kwargs['max_grad_norm'] = args.max_grad_norm
        dp_sgd_params = DpSgdParameters(**kwargs)

    training_params = TrainingParameters(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_dp=args.use_dp,
        dp_params=dp_sgd_params if args.use_dp else None
    )

    main(dataset_choice, args.data_path, training_params)

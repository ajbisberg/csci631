import argparse
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
import numpy as np
from os import makedirs
from os.path import join
import pickle
import torch

from data import DatasetChoice, load_data
from model import AdultModel, HousingModel
from plot_results import save_results_fig
from train import DpSgdParameters, train, TrainingParameters
from test import test


@dataclass
class MetaParameters:
    target_epsilons: list
    dataset_choice: DatasetChoice
    data_path: str
    num_trials: int = 1
    save_models: bool = False


def get_batch_size(meta_params):
    train_dataloader, cross_val_dataloader, test_dataloader = load_data(meta_params.dataset_choice, meta_params.data_path, 1)

    return int(np.sqrt(len(train_dataloader)))


def train_test(meta_params: MetaParameters, training_params: TrainingParameters):
    train_dataloader, cross_val_dataloader, test_dataloader = load_data(meta_params.dataset_choice, meta_params.data_path, training_params.batch_size)

    if meta_params.dataset_choice == DatasetChoice.HOUSING:
        model = HousingModel()
    elif meta_params.dataset_choice == DatasetChoice.ADULT:
        model = AdultModel()
    else:
        raise NotImplementedError('No model available for dataset choice.')

    model, threshold = train(model, train_dataloader, cross_val_dataloader, training_params)

    if meta_params.save_models:
        now_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        makedirs('models', exist_ok=True)
        torch.save(model, join('models', 'model_' + now_str + '.pickle'))

    return test(meta_params.dataset_choice, model, threshold, test_dataloader)


def main(meta_params: MetaParameters, training_params: TrainingParameters):
    training_params.batch_size = get_batch_size(meta_params)
    print('Using batch size', training_params.batch_size)

    res = {}
    if training_params.use_dp:
        for target_epsilon in meta_params.target_epsilons:
            training_params.dp_params.target_epsilon = target_epsilon
            with Pool(max(meta_params.num_trials, cpu_count() - 2)) as pool:
                agg_results = pool.starmap(train_test, [(meta_params, training_params)] * meta_params.num_trials)
            results = np.mean(agg_results, axis=0)
            res[target_epsilon] = results

    now_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    makedirs('results', exist_ok=True)

    name = str(meta_params.dataset_choice) + '_num_trials' + str(meta_params.num_trials) + '_' + now_str

    with open(join('results', 'results_' + name + '.pickle'), 'wb') as f:
        pickle.dump(res, f)

    makedirs('figs', exist_ok=True)
    save_results_fig(join('figs', 'fig_' + name), res)


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
    parser.add_argument('--target-epsilons',
                        help='Target epsilons for DP-SGD',
                        type=float,
                        nargs='+',
                        action='extend')
    parser.add_argument('--target-delta',
                        help='Target delta for DP-SGD',
                        type=float,
                        default=1e-5)
    parser.add_argument('--max-grad-norm',
                        help='Max gradient norm for DP-SGD',
                        type=float,
                        default=1.0)
    parser.add_argument('--num-trials',
                        help='Number of trials to average results across',
                        type=int,
                        default=1)
    parser.add_argument('--save-models',
                        help='Write trained models to the disk',
                        action='store_true')
    args = parser.parse_args()


    if args.use_dp:
        kwargs = {
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

    dataset_choice = DatasetChoice[args.dataset]

    meta_parameters = MetaParameters(
        dataset_choice=dataset_choice,
        target_epsilons=args.target_epsilons,
        num_trials=args.num_trials,
        data_path=args.data_path,
        save_models=args.save_models
    )

    main(meta_parameters, training_params)

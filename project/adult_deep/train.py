from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from opacus import PrivacyEngine
import torch
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, RocCurveDisplay, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DpSgdParameters:
    """Class for DP-SGD parameters"""
    target_delta: float
    target_epsilon: float = None
    max_grad_norm: float = 1.0


@dataclass
class TrainingParameters:
    """Class for deep learning training parameters"""
    num_epochs: int
    batch_size: int
    learning_rate: float = 1e-4
    use_dp: bool = False
    dp_params: DpSgdParameters = None


def cross_validate(model, cross_val_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        all_predictions, all_labels = [], []
        for i, (rows, labels) in tqdm(enumerate(cross_val_loader)):
            predictions = model(rows)
            all_predictions.append(predictions)
            all_labels.append(labels)
            correct += int(sum(predictions.round() == labels))
            total += len(labels)

        predictions = torch.vstack(all_predictions)
        labels = torch.vstack(all_labels)

        print('Cross Val Accuracy: %.3f' % (correct / total))
        score = roc_auc_score(labels, predictions)

        print('Cross Val ROC AUC Score: %.3f' % score)

        fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print('Cross Val Threshold: %.3f' % optimal_threshold)

        tresholded_accuracy = accuracy_score(predictions > optimal_threshold, labels == 1)

        print('Cross Val Thresholded Accuracy: %.3f' % tresholded_accuracy)

    return optimal_threshold


def train(model, train_loader, cross_val_loader, training_params):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=training_params.learning_rate)

    privacy_engine = PrivacyEngine()

    if training_params.use_dp:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=training_params.num_epochs,
            max_grad_norm=training_params.dp_params.max_grad_norm,
            target_epsilon=training_params.dp_params.target_epsilon,
            target_delta=training_params.dp_params.target_delta)

    running_loss = 0
    for epoch in range(training_params.num_epochs):
        for i, (rows, labels) in tqdm(enumerate(train_loader)):
            model.train()
            optimizer.zero_grad()
            predictions = model(rows)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if i == len(train_loader) - 1:
                accuracy = accuracy_score(predictions.round().detach(), labels)
                print('Epoch %d, Loss: %.3f, Training Accuracy: %.3f' %
                      (epoch + 1, running_loss / training_params.batch_size, accuracy))
                if training_params.use_dp:
                    print('Privacy Budget Exhausted: %5f' % privacy_engine.get_epsilon(delta=training_params.dp_params.target_delta))
                running_loss = 0

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for i, (rows, labels) in tqdm(enumerate(cross_val_loader)):
                predictions = model(rows)
                correct += int(sum(predictions.round() == labels))
                total += len(labels)
            print('Cross Val Accuracy: %.3f' % (correct / total))

    optimal_threshold = cross_validate(model, cross_val_loader)

    return model, optimal_threshold

import numpy as np
from opacus import PrivacyEngine
import torch
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, cross_val_loader, batch_size=64, use_dp=False, ):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)

    privacy_engine = PrivacyEngine()

    if use_dp:
        target_delta = 10e-5
        num_epochs = 10
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            max_grad_norm=1.0,
            target_epsilon=0.1,
            target_delta=target_delta,
            epochs = num_epochs
        )

    running_loss = 0
    for epoch in range(num_epochs):
        for i, (rows, labels) in enumerate(train_loader):
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
                      (epoch + 1, running_loss / batch_size, accuracy))
                if use_dp:
                    print('Privacy Budget Exhausted: %5f' % privacy_engine.get_epsilon(delta=target_delta))
                running_loss = 0

        if cross_val_loader:
            with torch.no_grad():
                correct, total = 0, 0
                for i, (rows, labels) in enumerate(cross_val_loader):
                    predictions = model(rows)
                    correct += int(sum(predictions.round() == labels))
                    total += len(labels)
                print('Cross Val Accuracy: %.3f' % (correct / total))

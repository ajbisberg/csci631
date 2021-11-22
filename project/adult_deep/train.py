from opacus import PrivacyEngine
from torch import optim, nn
from tqdm import tqdm


def train(model, data_loader, batch_size=128, use_dp=False):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if use_dp:
        privacy_engine = PrivacyEngine(
            model,
            alphas=[10, 100],
            noise_multiplier=1.3,
            max_grad_norm=1.0,
            batch_size=batch_size,
            sample_size=len(data_loader)
        )
        privacy_engine.attach(optimizer)

    running_loss = 0
    for epoch in range(10):
        for i, (rows, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            predictions = model(rows)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i == batch_size - 1:
                misclassification_rate = sum(abs(predictions.round() - labels))/len(predictions)
                print('Epoch %d, Loss: %.3f, Misclassification: %.3f' %
                      (epoch + 1, running_loss / batch_size, misclassification_rate))
                running_loss = 0

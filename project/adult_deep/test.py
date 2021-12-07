from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm

import constants
from data import DatasetChoice, load_data


def predict():
    all_p, all_labels = []


def get_gender_index(dataset_choice):
    if dataset_choice == DatasetChoice.HOUSING:
        return constants.HOUSING_GENDER_MALE_COLUMN_IDX
    elif dataset_choice == DatasetChoice.ADULT:
        return constants.ADULT_GENDER_MALE_COLUMN_IDX


def get_race_index(dataset_choice):
    if dataset_choice == DatasetChoice.HOUSING:
        return constants.HOUSING_RACE_WHITE_NON_HISPANIC_COLUMN_IDX
    elif dataset_choice == DatasetChoice.ADULT:
        return constants.ADULT_RACE_WHITE_COLUMN_IDX


def test(dataset_choice, model, threshold, dataloader):
    model.eval()
    with torch.no_grad():
        all_rows, all_predictions, all_labels = [], [], []
        for rows, labels in tqdm(dataloader):
            predictions = model(rows)
            predictions = (predictions > threshold).float()
            all_rows.append(rows)
            all_predictions.append(predictions)
            all_labels.append(labels)

        rows = torch.vstack(all_rows).numpy()
        predictions = torch.vstack(all_predictions).numpy()
        labels = torch.vstack(all_labels).numpy()

        gender_idx = get_gender_index(dataset_choice)
        race_idx = get_race_index(dataset_choice)

        race = rows[:, race_idx]
        gender = rows[:, gender_idx]

        return (
            accuracy_score(labels, predictions),
            demographic_parity_ratio(labels, predictions, sensitive_features=gender),
            demographic_parity_ratio(labels, predictions, sensitive_features=race),
            equalized_odds_ratio(labels, predictions, sensitive_features=gender),
            equalized_odds_ratio(labels, predictions, sensitive_features=race))

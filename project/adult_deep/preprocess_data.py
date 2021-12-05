import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline


FEATURES=[
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'salary'
]

RACES=[
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black"
]

MARITAL_STATUSES=[
    "Married-civ-spouse",
    "Divorced",
    "Never-married",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

OCCUPATIONS=[
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
    "?"
]

RELATIONSHIPS=[
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

WORKCLASSES=[
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
    "?"
]

NATIVE_COUNTRIES=[
    "United-States",
    "Cambodia",
    "England",
    "Puerto-Rico",
    "Canada",
    "Germany",
    "Outlying-US(Guam-USVI-etc)",
    "India",
    "Japan",
    "Greece",
    "South",
    "China",
    "Cuba",
    "Iran",
    "Honduras",
    "Philippines",
    "Italy",
    "Poland",
    "Jamaica",
    "Vietnam",
    "Mexico",
    "Portugal",
    "Ireland",
    "France",
    "Dominican-Republic",
    "Laos",
    "Ecuador",
    "Taiwan",
    "Haiti",
    "Columbia",
    "Hungary",
    "Guatemala",
    "Nicaragua",
    "Scotland",
    "Thailand",
    "Yugoslavia",
    "El-Salvador",
    "Trinadad&Tobago",
    "Peru",
    "Hong",
    "Holand-Netherlands",
    "?"
]

SALARIES=[
    "<=50K",
    ">50K"
]

SEXES=[
    "Male",
    "Female"
]


def create_row_transformer(df):
    column_transformer = make_column_transformer(
        (OneHotEncoder(categories=[RACES], sparse=False), ['race']),
        (OneHotEncoder(categories=[MARITAL_STATUSES], sparse=False), ['marital_status']),
        (OneHotEncoder(categories=[RELATIONSHIPS], sparse=False), ['relationship']),
        (OneHotEncoder(categories=[WORKCLASSES], sparse=False), ['workclass']),
        (OneHotEncoder(categories=[NATIVE_COUNTRIES], sparse=False), ['native_country']),
        (OneHotEncoder(categories=[OCCUPATIONS], sparse=False), ['occupation']),
        (OneHotEncoder(categories=[SEXES], sparse=False, drop='if_binary'), ['sex']),
        remainder='passthrough')
    transformer = Pipeline([
        ('column_transformer', column_transformer),
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])
    transformer.fit(df)

    return transformer


def create_label_transformer(labels):
    transformer = LabelBinarizer()
    transformer.fit(labels)

    return transformer



def split_data(rows, labels, train_proportion=0.7,
               cross_val_proportion=0.15, test_proportion=0.15):
    assert(train_proportion + cross_val_proportion + test_proportion == 1)
    indices = np.arange(len(rows))
    np.random.shuffle(indices)

    train_cutoff = int(len(indices) * train_proportion)
    cross_val_cutoff = int(len(indices) * (train_proportion + cross_val_proportion))

    train_rows = rows[indices[:train_cutoff]]
    train_labels = labels[indices[:train_cutoff]]
    cross_val_rows = rows[indices[train_cutoff:cross_val_cutoff]]
    cross_val_labels = labels[indices[train_cutoff:cross_val_cutoff]]
    test_rows = rows[indices[cross_val_cutoff:]]
    test_labels = labels[indices[cross_val_cutoff:]]

    return train_rows, train_labels, cross_val_rows, cross_val_labels, test_rows, test_labels


def split_labels(df):
    labels = df[['salary']]
    rows = df.drop(columns=['education', 'salary'])

    return rows, labels


def main(train_data_path, test_data_path):
    train_df = pd.read_csv(train_data_path, names=FEATURES, skipinitialspace=True)
    test_df = pd.read_csv(test_data_path, names=FEATURES, skipinitialspace=True, skiprows=1)

    df = pd.concat([train_df, test_df])

    rows, labels = split_labels(df)
    labels = labels.replace('<=50K.', '<=50K').replace('>50K.', '>50K')

    row_transformer = create_row_transformer(rows)
    label_transformer = create_label_transformer(labels)

    rows = row_transformer.transform(rows)
    labels = label_transformer.transform(labels)

    train_rows, train_labels, cross_val_rows, cross_val_labels, test_rows, test_labels = split_data(rows, labels)

    with open('adult.pickle', 'wb') as f:
        data = {
            'row_transformer': row_transformer,
            'label_transformer': label_transformer,
            'train_rows': train_rows,
            'train_labels': train_labels,
            'cross_val_rows': cross_val_rows,
            'cross_val_labels': cross_val_labels,
            'test_rows': test_rows,
            'test_labels': test_labels,
        }
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                      description='Re-split Adult dataset')
    parser.add_argument('--train-data-path',
                        help='Path to Adult training data (CSV)',
                        required=True)
    parser.add_argument('--test-data-path',
                        help='Path to Adult test data (CSV)',
                        required=True)
    args = parser.parse_args()

    main(args.train_data_path, args.test_data_path)

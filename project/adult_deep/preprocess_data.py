import pandas as pd
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
    transformer.classes_ = ['<=50K', '>50K']
    transformer.fit(labels)

    return transformer


def load_data(fname):
    df = pd.read_csv(fname, names=FEATURES, skipinitialspace=True)

    labels = df[['salary']]
    rows = df.drop(columns=['education', 'salary'])

    row_transformer = create_row_transformer(rows)
    label_transformer = create_label_transformer(labels)

    transformed_rows = row_transformer.transform(rows)
    transformed_labels = label_transformer.transform(labels)

    return transformed_rows, transformed_labels

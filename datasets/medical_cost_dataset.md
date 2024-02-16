# Medical Cost Personal Dataset

## Overview
This dataset provides medical cost information for 1338 individuals in a CSV format, which can be used for regression analysis tasks. For more information, see the dataset on [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).

## Features
The dataset includes the following attributes for each person:

1. `age`: The age of the individual.
2. `sex`: The biological sex of the individual.
3. `bmi`: Body Mass Index, which provides an understanding of body, weights that are relatively high or low relative to height.
4. `children`: The number of children/dependents covered by health insurance.
5. `smoker`: Smoking status of the individual.
6. `region`: The beneficiary's residential area in the US, divided into four geographic regions - northeast, southeast, southwest, or northwest.
7. `charges`: Individual medical costs billed by health insurance (the target variable for prediction).

## Preprocessing
The `sex` and `smoker` attributes have been converted into binary features for analysis purposes, effectively one-hot encoding these categories. The `region` feature has been transformed into a set of binary features as well. The `charges` data is included and will be the target variable for predictive modeling.

---

For instructions on how to access and use this dataset in your analyses, refer to the main project `README.md`.

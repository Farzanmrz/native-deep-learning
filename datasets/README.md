# Kid Creative Dataset

The Kid Creative dataset is intended for binary classification tasks. It contains demographic and socioeconomic information for 673 individuals, recorded in a CSV file. This dataset is particularly useful for predicting marketing outcomes, such as the likelihood of a customer to purchase a product.

The data for each person includes the following attributes:

1. `Buy`: Whether the individual made a purchase (binary target value, Y).
2. `Income`: The individual's income level.
3. `Is Female`: Indicates if the individual is female.
4. `Is Married`: Indicates if the individual is married.
5. `Has College`: Indicates if the individual has a college degree.
6. `Is Professional`: Indicates if the individual is a professional.
7. `Is Retired`: Indicates if the individual is retired.
8. `Unemployed`: Indicates if the individual is unemployed.
9. `Residence Length`: The length of time the individual has lived at their residence.
10. `Dual Income`: Indicates if the household has a dual income.
11. `Minors`: Indicates if there are minors (children) in the household.
12. `Own`: Indicates if the individual owns their residence.
13. `House`: Indicates if the individual lives in a house.
14. `White`: Indicates if the individual is white.
15. `English`: Indicates if the individual primarily speaks English.
16. `Prev Child Mag`: Indicates if the individual previously purchased a children's magazine.
17. `Prev Parent Mag`: Indicates if the individual previously purchased a parenting magazine.

The first column, 'Observation Number', will be omitted from analyses as it does not provide predictive value. The second column, 'Buy', will serve as our binary target variable Y. The remaining 16 columns will constitute our feature data for the observation matrix X.

# Medical Cost Personal Dataset

This dataset provides medical cost information for 1338 individuals in a CSV format, which can be used for regression analysis tasks. For more information, see the dataset on [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).

The dataset includes the following attributes for each person:

1. `age`: The age of the individual.
2. `sex`: The biological sex of the individual.
3. `bmi`: Body Mass Index, which provides an understanding of body, weights that are relatively high or low relative to height.
4. `children`: The number of children/dependents covered by health insurance.
5. `smoker`: Smoking status of the individual.
6. `region`: The beneficiary's residential area in the US, divided into four geographic regions - northeast, southeast, southwest, or northwest.
7. `charges`: Individual medical costs billed by health insurance (the target variable for prediction).

The `sex` and `smoker` attributes have been converted into binary features for analysis purposes, effectively one-hot encoding these categories. The `region` feature has been transformed into a set of binary features as well. The `charges` data is included and will be the target variable for predictive modeling.
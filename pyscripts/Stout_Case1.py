# %%
# Importing libraries and initial configs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    PredefinedSplit,
)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor


from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100, "display.max_rows", 100)


# %% [markdown]
# ### Case Study #1

# %%
df = pd.read_csv("./data/loans_full_schema.csv")
df.head()


# %% [markdown]
# #### *The dataset - descriptions, overview, issues*

# %%
# Variables and their data types
df.info()


# %%
# Descriptive statistics of numerical attributes
df.describe()


# %%
# Checking for duplicates
df.duplicated().any()

# %%
# Checking the proportion of nulls in the dataframe
nulls = pd.DataFrame(
    {
        "Null Values": df.isna().sum(),
        "Percentage Null Values": (df.isna().sum() / len(df)).apply(
            lambda x: "{:.2%}".format(x)
        ),
    }
)
nulls


# %%
# Count of unique values for loan term
df["term"].value_counts()


# %%
# Count of unique values for num_accounts_120d_past_due
df["num_accounts_120d_past_due"].value_counts()

# %%
# Count of unique values for num_accounts_30d_past_due
df["num_accounts_30d_past_due"].value_counts()

# %%
# Count of unique values for current_accounts_delinq
df["current_accounts_delinq"].value_counts()

# %% [markdown]
# The dataset contains records of loans that were offered and taken up through Lending Club. It has 10000 unique observations, each corresponding to one loan taken up. It has 55 features corresponding to various attributes related to the credit risk of a customer.
#
# The loan terms in this dataset have two unique values, thus only 36 and 60 month loan terms were offered. The attribute `num_accounts_120d_past_due` in this dataset has only values populated 0, or nulls. Both `num_accounts_30d_past_due` and `current_accounts_delinq` also do not have any variation in the unique values. `annual_income_joint`, `verification_income_joint`, `debt_to_income_joint`, `months_since_last_delinq` and `months_since_90d_late` have very prpoportions of nulls.

# %% [markdown]
# #### *Visualizations - Exploratory analysis*

# %%
sns.set_style("darkgrid")

# Checking the distribution of the interest rates
sns.displot(data=df, x="interest_rate", bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.title("Distribution of interest rates", fontsize=15)
plt.show()


# %%
# Checking the distribution of the log of interest rates
sns.displot(np.log(df["interest_rate"]), bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.title("Distribution of log of interest rates", fontsize=15)
plt.show()


# %% [markdown]
# Looking at the distribution of interest rates, we observe that the distribution is skewed to the right. We also observe the log transformation of this variable which is less skewed, but not perfectly normally distributed.

# %%
# Checking the distribution of interest rates by grade
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="grade",
    y="interest_rate",
    order=["A", "B", "C", "D", "E", "F", "G"],
    palette="Set2",
)
plt.xlabel("Grade of loan")
plt.ylabel("Interest Rate")
plt.title("Distribution of interest rates by grade", fontsize=15)
plt.show()


# %% [markdown]
# The grade of the loan, from 'A' to 'G', shows a nicely linear relationship with the interest rate, as can be seen from the graph above.

# %%
# Comparing debt_to_income and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="debt_to_income",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Debt to income ratio")
plt.ylim(0, 80)
plt.ylabel("Interest Rate")
plt.title("Interest rate vs Debt to income ratio", fontsize=15)
plt.show()


# %% [markdown]
# Debt-to-income is a good indicator of the financial health of an individual, with higher values signalling more risk. While most values are within the 100% threshold, we observe some values where individuals have potentially more debt than income. The interest rate slopes up with this attribute.

# %%
# Comparing account_never_delinq_percent and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="account_never_delinq_percent",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Accounts never delinquent percentage")
plt.ylabel("Interest Rate")
plt.ylim(0, 80)
plt.title("Interest rate vs Account never delinquent percentage", fontsize=15)
plt.show()


# %% [markdown]
# The percentage of never delinquent accounts is another good indicator for credit risk. Here we observe that the higher the number of never delinquent accounts held by an individual, the lower interest rate he gets offered.

# %%
# Comparing accounts_opened_24m and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="accounts_opened_24m",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Accounts opened in last 24 months")
plt.ylabel("Interest Rate")
plt.ylim(0, 80)
plt.title("Interest rate vs Accounts opened in last 24 months", fontsize=15)
plt.show()


# %% [markdown]
# The higher the number of accounts opened in the last 24 months, could be an indicator of how credit-hungry an individual is. The trend for interest rates slopes up with this attribute, as can be seen in the plot above.

# %%
# Comparing delinq_2y and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="delinq_2y",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Number of delinquencies in last 2 years")
plt.ylabel("Interest Rate")
plt.ylim(0, 80)
plt.title("Interest rate vs Number of delinquencies in last 2 years", fontsize=15)
plt.show()


# %% [markdown]
# If there are a higher number of delinquencies reported against an individual in the preceding two years, it could be an indicator of higher risk. The interest rates tend to go up when this increases.

# %%
# Comparing num_cc_carrying_balance and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="num_cc_carrying_balance",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Number of credit cards with balance")
plt.ylabel("Interest Rate")
plt.ylim(0, 80)
plt.title("Interest rate vs Number of credit cards with balance", fontsize=15)
plt.show()


# %% [markdown]
# The number of credit cards carrying balance is a good indicator of active credit usage by a customer. Higher amounts may indicate higher risk, and hence, we observe that interest rates increase with higher numbers of actively used credit cards.

# %%
# Comparing annual_income and interest_rate
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x="annual_income",
    y="interest_rate",
    line_kws={"color": "orange"},
    fit_reg=True,
)
plt.xlabel("Annual income")
plt.ylabel("Interest Rate")
plt.ylim(0, 80)
plt.title("Interest rate vs Annual income", fontsize=15)
plt.show()


# %% [markdown]
# We observe that the interest rates decerease with higher values of income, which is to be expected, as offering loans to higher income inviduals is not risky.

# %%
# Checking the distribution of loan purpose by grade
plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x="loan_purpose", y="interest_rate", palette="Set2")
plt.xlabel("Purpose of loan")
plt.xticks(rotation=45)
plt.ylabel("Interest Rate")
plt.title("Distribution of interest rates by loan purpose", fontsize=15)
plt.show()


# %% [markdown]
# We observe here that there is significant variation in the interest rates by the purpose for which the loan is taken. Mortgages and loans for medical purposes have low interest rates, while loans for businesses have higher rates.

# %% [markdown]
# #### *Pre-processing the dataset*

# %%
# Dropping columns populated with a single value throughout
df.drop(
    columns=[
        "num_accounts_120d_past_due",
        "num_accounts_30d_past_due",
        "current_accounts_delinq",
    ],
    inplace=True,
)


# %%
# Dropping columns where proportion of nulls is greater than 50%
df.drop(
    columns=nulls.loc[nulls["Null Values"] / len(df) >= 0.5].index.tolist(),
    inplace=True,
)
print(
    "Dataframe shape after dropping columns with more than 50% null values:", df.shape
)


# %%
# Separating features into categorical and numerical attributes
df_cat = df.select_dtypes(include=["object"]).copy()
df_num = df.select_dtypes(include=["int64", "float64"]).copy()


# %%
# Using mode imputation to impute nulls for the categorical features
df_cat.fillna(df_cat.mode().iloc[0], inplace=True)


# %%
# Using multiple imputation to impute nulls for the numeric features
# This estimates the nulls from the other features, it models each feature with missing values as a function of other features

imputer = IterativeImputer(random_state=0)
df_impute = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
df_impute.head()


# %%
# Checking for presence of collinearity among the numerical features
fig = plt.figure(figsize=(20, 18))

corr = df_impute.corr()
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    # annot=True,
    # fmt=".2%",
    cmap="YlGnBu",
    mask=np.triu(corr),
)
plt.title("Correlation between numerical features", fontsize=15)
plt.show()


# %%
# corr.to_csv("./corr.csv")

# %%
corr_cols_to_drop = [
    "total_credit_lines",
    "num_satisfactory_accounts",
    # "num_accounts_30d_past_due",
    "num_active_debit_accounts",
    "num_open_cc_accounts",
    "num_total_cc_accounts",
    "tax_liens",
    "installment",
    "balance",
    "paid_interest",
    "paid_late_fees",
    "paid_principal",
    "paid_total",
    "num_cc_carrying_balance",
]


# %%
df_subset = df_impute.drop(columns=corr_cols_to_drop)
print("Dataframe shape after dropping columns with high correlation:", df_subset.shape)

# %% [markdown]
# An alternate visualization can be found in the file `./artifacts/corr.xlsx`. A threshold of 70% is used to determine which correlated features to drop. For similar pairs like `total_credit_lines` and `open_credit_lines`, or the attribute corresponding to `open_*` is shortlisted as that is a more accurate indicator of the active credit behaviour of the individual. All attributes which correspond to behaviour of the loan post its disbursement (`paid_principal`, `balance`, `paid_total`, etc.) are dropped, as through this analysis, we seek to determine if it is possible to develop a predictive model that can accurately price a loan, given various credit attributes of an individual.

# %%
# Re-checking correlation heatmap
fig = plt.figure(figsize=(20, 18))
corr2 = df_subset.corr()
sns.heatmap(
    corr2,
    xticklabels=corr2.columns.values,
    yticklabels=corr2.columns.values,
    # annot=True,
    # fmt=".2%",
    cmap="YlGnBu",
    mask=np.triu(corr2),
)
plt.title("Correlation between shortlisted numerical features", fontsize=15)
plt.show()


# %%
# Pre-processing on the categorical features
df_cat["emp_title"].value_counts().index[:30]


# %% [markdown]
# We observe here that the numer of unique values corresponding to this attribute is very high, often with variations of the same title. To avoid extremely high dimensionality while modeling, we drop this variable from further consideration. Some additonal categorical features listed below are also dropped, as these do not add any value in the prediction of interest rates.

# %%
cat_cols_to_drop = [
    "emp_title",
    "issue_month",
    "loan_status",
    "disbursement_method",
    "sub_grade",
    "state",
]


# %%
df_cat.drop(columns=cat_cols_to_drop, inplace=True)
df_cat.shape


# %%
df_cat.head()


# %%
# Using one-hot encoding for the shortlisted categorical features

ohe = OneHotEncoder(handle_unknown="ignore")
df_cat_ohe = pd.DataFrame(
    ohe.fit_transform(df_cat).toarray(), columns=ohe.get_feature_names_out()
)
df_cat_ohe.head()


# %%
# Merging back the pre-processed numerical and categorical attributes together
df_full = pd.concat([df_subset, df_cat_ohe], axis=1)
print(
    "Dataframe shape after merging numerical and categorical features:", df_full.shape
)


# %%
# Re-checking any null values
df_full.isna().any().any()


# %%
# Train-test split
train, test = train_test_split(df_full, test_size=0.3, random_state=0)
X_train = train.drop(columns=["interest_rate"])
y_train = train["interest_rate"]
X_test = test.drop(columns=["interest_rate"])
y_test = test["interest_rate"]


# %%
# Further splitting train dataset to training and validation
np.random.seed(8642)

mask = np.random.rand(X_train.shape[0]) < 0.8

X_tr = X_train[mask]
X_val = X_train[~mask]

y_tr = y_train[mask]
y_val = y_train[~mask]


# %%
# Create a predefined train/test split for GridSearchCV (to be used later)
validation_fold = np.concatenate((-1 * np.ones(len(y_tr)), np.zeros(len(y_val))))
train_val_split = PredefinedSplit(validation_fold)


# %% [markdown]
# ##### Fitting Linear Regression

# %%
# Fitting linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print("On training set: {:.2%}".format(lin_reg.score(X_train, y_train)))
print("On test set: {:.2%}".format(lin_reg.score(X_test, y_test)))


# %%
# Histogram of residuals
y_train_pred = lin_reg.predict(X_train)
residuals = y_train - y_train_pred

sns.displot(residuals, bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Residuals")
plt.title("Residuals of training set")
plt.show()

# %%
# Visualizing residuals
visualizer = ResidualsPlot(lin_reg, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# %%
# Fitting linear regression with log transform of the target variable
log_lin_reg = LinearRegression()
log_lin_reg.fit(X_train, np.log(y_train))

print("On training set: {:.2%}".format(log_lin_reg.score(X_train, np.log(y_train))))
print("On test set: {:.2%}".format(log_lin_reg.score(X_test, np.log(y_test))))


# %%
# Visualizing residuals
y_train_pred = log_lin_reg.predict(X_train)
residuals = np.log(y_train) - y_train_pred

sns.displot(residuals, bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Residuals with log transformed target variable")
plt.title("Residuals of training set")
plt.show()

# %%
visualizer = ResidualsPlot(log_lin_reg, hist=False, qqplot=True)
visualizer.fit(X_train, np.log(y_train))
visualizer.score(X_test, np.log(y_test))
visualizer.show()


# %% [markdown]
# ##### Fitting linear regression with Lasso regularization

# %%
# Fitting Lasso
lass = Lasso(random_state=0)
lass.fit(X_train, y_train)

print("On training set: {:.2%}".format(lass.score(X_tr, y_tr)))
print("On training set: {:.2%}".format(lass.score(X_val, y_val)))
print("On test set: {:.2%}".format(lass.score(X_test, y_test)))


# %%
# Performing grid search to find the best hyperparameters
param_grid = {
    "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    "selection": ["cyclic", "random"],
}

lasso_cv = Lasso(random_state=0, max_iter=10000)
gridsearch = GridSearchCV(
    lasso_cv,
    cv=train_val_split,
    param_grid=param_grid,
    # n_iter=10000,
    scoring="r2",
    n_jobs=-2,
    # random_state=2,
)
search = gridsearch.fit(X_train, y_train)
print(
    "The best learning parameters obtained out of the randomized search are: \n",
    search.best_params_,
)
print(
    "The best score obtained out of the randomized search is: \n",
    search.best_score_,
)


# %%
# Fitting model again with best parameters obtained from grid search
lass = Lasso(random_state=0, max_iter=10000, alpha=0.001, selection="cyclic")
lass.fit(X_train, y_train)

print("On training set: {:.2%}".format(lass.score(X_train, y_train)))
# print("On validation set: {}".format(lass.score(X_val, y_val)))
print("On test set: {:.2%}".format(lass.score(X_test, y_test)))


# %%
y_train_pred = lass.predict(X_train)
residuals = y_train - y_train_pred

sns.displot(residuals, bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Residuals with log transformed target variable")
plt.title("Residuals of training set")
plt.show()

# %%
visualizer = ResidualsPlot(lass, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# %%
# Fitting lasso with log transform of the target variable, with hyperparameters obtained from grid search
log_lass = Lasso(random_state=0, max_iter=10000, alpha=0.001, selection="random")
log_lass.fit(X_train, np.log(y_train))

print("On training set: {:.2%}".format(log_lass.score(X_train, np.log(y_train))))
# print("On validation set: {}".format(lass.score(X_val, y_val)))
print("On test set: {:.2%}".format(log_lass.score(X_test, np.log(y_test))))


# %%
y_train_pred = log_lass.predict(X_train)
residuals = np.log(y_train) - y_train_pred

sns.displot(residuals, bins=50, kde=True, height=5, aspect=1.5)
plt.xlabel("Residuals with log transformed target variable")
plt.title("Residuals of training set")
plt.show()

# %%
visualizer = ResidualsPlot(log_lass, hist=False, qqplot=True)
visualizer.fit(X_train, np.log(y_train))
visualizer.score(X_test, np.log(y_test))
visualizer.show()


# %% [markdown]
# As can be seen from the plots above, the following iterations are performed:
#
# 1. Linear Regression
# 1. Linear Regression with log transformed target
# 1. Linear Regression with Lasso regularization
# 1. Linear Regression with Lasso regularization and with log transformed target
#
# Both Lasso models are fit using tuned parameters obtained from the grid search algorithm. The grid search algorithm cross validates concurrently on the validation dataset to decide the model with the best performance. The metric used to decide between models is the R-squared metric, which explains the amount of variance explained by the model. Once the tuned parameters are obatined, the model is refit on all the training and validation data and tested on the test dataset.
#
# For iterations (1) and (3), we observe that the distribution of the residuals is very skewed. This is further confirmed by the QQ plot, which leads us to conclude the the assumption of normality of residuals is violated. Further, looking at residual plots, we can also observe that the assumption of independence of observations is also violated.
#
# For iterations (2) and (4), we also observe that the assumption of normality of residuals is violated, from the distribution of residuals as well as the QQ plots. Further, in these iterations, we observe that the assumptions of independence of observations, as well as the assumptions of homoscedasticity are violated, as can be observed by the presence of patterns amongst the residuals.
#
# Thus, the linear regression model results tell us that the assumptions are violated and hence, should not be used for making decisions.
#
#
#
#

# %% [markdown]
# ##### Fitting a Random Forest Regressor

# %%
# Fitting random forest
rf = RandomForestRegressor(n_estimators=1000, max_depth=25, max_features=3)
rf.fit(X_train, y_train)

print("On training set: {:.2%}".format(rf.score(X_train, y_train)))
print("On test set: {:.2%}".format(rf.score(X_test, y_test)))


# %%
importance = rf.feature_importances_
plt.figure(figsize=(12, 18))

# plot feature importance
plt.barh(X_train.columns, importance)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Feature importances from Random Forest Regressor", fontsize=15)
plt.show()


# %% [markdown]
# We observe that the Random Forest Regressor has achieves an R-squared score of ~88% on the test dataset. Grade, loan term and loan amount can be observed as some of the most important features, as observed from the plot above. Since this algorithm uses multiple trees and averages out the prediction value for the target, the bias can be greatly reduced. However, this regressor can quickly get complex if we increase the number of trees.

# %% [markdown]
# ##### Limitations and Future Work

# %% [markdown]
# One-hot encoding leads to increased dimensionality of the dataset, resulting in sparse matrices of coefficients. Tree-based algorithms are a good way to deal with this, and different tree-based regressors with bagging can be performed to check for better model performance.
#
# Scaling and normalization of features can be carried out, although tree-based algorithms work well without scaling. Outliers need to be properly analyzed, since these can significantly affects models like linear regression.
#
# For the `emp_title` attribute, we observed that there are many categories which are similar, for example, "registered nurse", "rn", and "nurse" could be collapsed into a single category. A simple NLP algorithm which leverages Levenshtein distances could be used to this end to reduce the number of unique values of this attribute.

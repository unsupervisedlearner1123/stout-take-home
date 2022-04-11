# %%
# Importing libraries and initial configs
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 100, "display.max_rows", 100)


# %% [markdown]
# ### Case Study #2

# %%
df = pd.read_csv("./data/casestudy.csv").iloc[:, 1:]


# %%
df.head()


# %%
df.shape


# %%
df.duplicated(["customer_email", "year"]).any()


# %% [markdown]
# #### Total revenue for the current year

# %%
df.groupby("year")["net_revenue"].sum()


# %% [markdown]
# #### New Customer Revenue e.g. new customers not present in previous year only

# %%
def calc_new_rev(prev_year, curr_year):
    """
    Calculates net revenue for new customers, not present in previous year
        Inputs:
            prev_year: Previous year
            curr_year: Current year
        Output:
            Net revenue for new customers in current year
    """
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    new_cust = cust_curr.difference(cust_prev)
    return df.iloc[
        np.where(
            np.logical_and(df["customer_email"].isin(new_cust), df["year"] == curr_year)
        )[0],
        :,
    ]["net_revenue"].sum()


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The net revenue from new customers in {} not present in {} is : {:.2f}".format(
            j, i, calc_new_rev(i, j)
        )
    )


# %% [markdown]
# #### Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year

# %%
def calc_exist_rev(curr_year, prev_year):
    """
    Calculates growth in revenue for existing customers in current year from previous year
        Inputs:
            curr_year: Current year
            prev_year: Previous year
        Output:
            Growth in revenue for existing customers in current year from previous year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    exist_cust = cust_prev.intersection(cust_curr)

    return (
        df.iloc[
            np.where(
                np.logical_and(
                    df["customer_email"].isin(exist_cust), df["year"] == curr_year
                )
            )[0],
            :,
        ]["net_revenue"].sum()
        - df.iloc[
            np.where(
                np.logical_and(
                    df["customer_email"].isin(exist_cust), df["year"] == prev_year
                )
            )[0],
            :,
        ]["net_revenue"].sum()
    )


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The net revenue in {} from existing customers in {} is : {:.2f}".format(
            j, i, calc_exist_rev(j, i)
        )
    )


# %% [markdown]
# #### Revenue lost from attrition

# %%
def calc_att_rev(curr_year, prev_year):
    """
    Calculates revenue lost from attrition in current year from previous year
        Inputs:
            curr_year: Current year
            prev_year: Previous year
        Output:
            Revenue lost from attrition in current year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    exist_cust = cust_prev.difference(cust_curr)

    return df.iloc[
        np.where(
            np.logical_and(
                df["customer_email"].isin(exist_cust), df["year"] == prev_year
            )
        )[0],
        :,
    ]["net_revenue"].sum()


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The revenue lost from attrition in {} is : {:.2f}".format(
            j, calc_att_rev(j, i)
        )
    )


# %% [markdown]
# #### Existing Customer Revenue Current Year

# %%
def calc_exist_rev_current(curr_year, prev_year):
    """
    Calculates revenue for existing customers in current year
        Inputs:
            curr_year: Current year
            prev_year: Previous year
        Output:
            Revenue for existing customers in current year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    exist_cust = cust_prev.intersection(cust_curr)

    return df.iloc[
        np.where(
            np.logical_and(
                df["customer_email"].isin(exist_cust), df["year"] == curr_year
            )
        )[0],
        :,
    ]["net_revenue"].sum()


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The revenue from existing customers in {} is : {:.2f}".format(
            j, calc_exist_rev_current(j, i)
        )
    )


# %% [markdown]
# #### Existing Customer Revenue Prior Year

# %%
def calc_exist_rev_prev(curr_year, prev_year):
    """
    Calculates revenue for existing customers in previous year
        Inputs:
            curr_year: Current year
            prev_year: Previous year
        Output:
            Revenue for existing customers in previous year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    exist_cust = cust_prev.intersection(cust_curr)

    return df.iloc[
        np.where(
            np.logical_and(
                df["customer_email"].isin(exist_cust), df["year"] == prev_year
            )
        )[0],
        :,
    ]["net_revenue"].sum()


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The revenue from existing customers in {} is : {:.2f}".format(
            i, calc_exist_rev_prev(j, i)
        )
    )


# %% [markdown]
# #### Total Customers Current Year

# %%
def calc_cust_curr(curr_year):
    """
    Calculates number of customers in current year
        Inputs:
            curr_year: Current year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])

    return len(cust_curr)


# %%
for i in [2015, 2016, 2017]:
    print("The number of customers in {} is : {:,}".format(i, calc_cust_curr(i)))


# %% [markdown]
# #### Total Customers Previous Year

# %%
def calc_cust_prev(curr_year):
    """
    Calculates number of customers in previous year
        Inputs:
            curr_year: Current year
    """
    if curr_year == 2015:
        return 0
    else:
        cust_prev = set(df.loc[df["year"] == curr_year - 1]["customer_email"])

    return len(cust_prev)


# %%
for i in [2015, 2016, 2017]:
    if i == 2015:
        print(
            "Current year is {}, number of customers in the previous year is : {:,}".format(
                i, calc_cust_prev(i)
            )
        )
    else:
        print(
            "Current year is {}, number of customers in the previous year {} is : {:,}".format(
                i, i - 1, calc_cust_prev(i)
            )
        )


# %% [markdown]
# #### New Customers

# %%
def calc_new_cust(prev_year, curr_year):
    """
    Calculates new customers in current year compared to previous year
        Inputs:
            prev_year: Previous year
            curr_year: Current year
        Output:
            Net revenue for new customers in current year
    """
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    new_cust = cust_curr.difference(cust_prev)
    return len(new_cust)


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The number of new customers in {} compared to {} is : {:.2f}".format(
            j, i, calc_new_cust(i, j)
        )
    )


# %% [markdown]
# #### Lost Customers

# %%
def calc_att_cust(curr_year, prev_year):
    """
    Calculates number of customers attrited in current year from previous year
        Inputs:
            curr_year: Current year
            prev_year: Previous year
    """
    cust_curr = set(df.loc[df["year"] == curr_year]["customer_email"])
    cust_prev = set(df.loc[df["year"] == prev_year]["customer_email"])
    exist_cust = cust_prev.difference(cust_curr)

    return len(exist_cust)


# %%
for (i, j) in zip(range(2015, 2017), range(2016, 2018)):
    print(
        "The number of customers attrited in {} from {} is : {:.2f}".format(
            j, i, calc_att_cust(i, j)
        )
    )

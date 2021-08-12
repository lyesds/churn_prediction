import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-muted')

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds = ds.iloc[:, 1:]  # remove IdClient

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

def piecharts(subtype: int = 0):
    categ = ds.select_dtypes(include='object').columns

    if subtype == 0:  # Target
        sns.set(font_scale=1.5)
        fig, axs = plt.subplots(1, 1, figsize=(7,7))
        axs = plt.pie(ds[categ[0]].value_counts(normalize=True).values,
                           labels=ds[categ[0]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
        plt.text(-1.5, .75, "m = 10127", fontsize=14, verticalalignment='top', bbox=props)
        plt.title('Target (attrition flag)')
    elif subtype == 1:  # social features
        sns.set(font_scale=.5)
        fig, axs = plt.subplots(3, 1, figsize=(20,10))
        for j in list(range(3)) :
            axs[j].title.set_text(categ[j+1])
            axs[j].pie(ds[categ[j+1]].value_counts(normalize=True).values,
                           labels=ds[categ[j+1]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
    elif subtype == 2:  # financial features
        sns.set(font_scale=.5)
        fig, axs = plt.subplots(2, 1, figsize=(20,10))
        for j in list(range(2)) :
            axs[j].title.set_text(categ[j+4])
            axs[j].pie(ds[categ[j+4]].value_counts(normalize=True).values,
                           labels=ds[categ[j+4]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
    return fig


def histos(numvarname: str = 'Customer_Age'):
    sns.set(font_scale=1)
    varlist = ds.select_dtypes(exclude='object').columns

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=.5)
    fig.suptitle(numvarname.replace('_', ' '))
    sns.histplot(ax=axs[0], x=ds[numvarname], kde=True)
    axs[0].title.set_text('Histogram (all samples)')
    sns.boxplot(ax=axs[1], y=numvarname, x='Attrition_Flag', data=ds)
    axs[1].title.set_text('Boxplot by attrition flag')
    axs[1].set_xlabel('')

    return fig, varlist

def scatters(varname1: str = 'Customer_Age', varname2: str = 'Total_Revolving_Bal'):
    sns.set(font_scale=1)
    varlist1 = ds.iloc[:, 1:].columns.to_list()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(varname2.replace('_', ' ')+' by '+varname1.replace('_', ' '))
    sns.scatterplot(ax=axs, x=varname1, y=varname2, hue='Attrition_Flag', data=ds)

    return fig, varlist1


# print(type(ds.iloc[:, 1:].columns.to_list()))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-muted')

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds = ds.iloc[:, 1:]  # remove IdClient

def piecharts(subtype: int):
    categ = ds.select_dtypes(include='object').columns

    if subtype == 0:  # Target
        sns.set(font_scale=1.5)
        fig, axs = plt.subplots(1, 1, figsize=(7,7))
        axs = plt.pie(ds[categ[0]].value_counts(normalize=True).values,
                           labels=ds[categ[0]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
        plt.title('Target (attrition flag)')
    elif subtype == 1:  # social features
        sns.set(font_scale=1)
        fig, axs = plt.subplots(3, 1, figsize=(20,10))
        for j in list(range(3)) :
            axs[j].title.set_text(categ[j+1])
            axs[j].pie(ds[categ[j+1]].value_counts(normalize=True).values,
                           labels=ds[categ[j+1]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
    elif subtype == 2:  # financial features
        sns.set(font_scale=1)
        fig, axs = plt.subplots(2, 1, figsize=(20,10))
        for j in list(range(2)) :
            axs[j].title.set_text(categ[j+4])
            axs[j].pie(ds[categ[j+4]].value_counts(normalize=True).values,
                           labels=ds[categ[j+4]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
    return fig


def histos(numvarname: str = 'Customer_Age'):
    sns.set(font_scale=2)
    varlist = ds.select_dtypes(exclude='object').columns

    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0] = sns.histplot(ds[numvarname], kde=True)
    plt.gca().update(dict(title='Histogram (all samples)', xlabel='', ylabel='Count'))
    axs[1] = sns.boxplot(data=ds, y=numvarname, x='Attrition_Flag')
    plt.title('Boxplot by attrition flag')

    return fig, varlist


'''varlist = ds.select_dtypes(exclude='object').columns
print(varlist)
'''
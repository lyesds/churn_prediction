import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot():
    ds = pd.read_csv('data/BankChurners.csv')
    ds = ds.iloc[:, :-2]  # discard two last columns
    categ = ds.select_dtypes(include='object').columns

    fig, axs = plt.subplots(2, 3, figsize=(20,10))
    for i in list(range(2)) :
        for j in list(range(3)) :
            axs[i , j].title.set_text(categ[j+i*3])
            axs[i , j].pie(ds[categ[j+i*3]].value_counts(normalize=True).values,
                           labels=ds[categ[j+i*3]].value_counts(normalize=True).index.values, startangle=90, autopct='%1.1f%%')
    plt.savefig("assets/pies_categ_variables.png",transparent=True)
    plt.show()
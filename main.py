import pathlib

import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd


# References:
# 1 - https://realpython.com/pandas-python-explore-dataset/

def get_dataset_path():
    directory = pathlib.Path().absolute()
    path = os.path.join(directory, 'dataset.csv')
    return path


if __name__ == '__main__':
    dataset_path = get_dataset_path()
    nba = pd.read_csv(dataset_path)

    # 1) Grandezza dataset
    size = nba.shape
    print("count row:" + str(size[0]))
    print("count columns:" + str(size[1]))

    # 2) Displaying Data Types
    v = nba.info()

    # 2) Displaying Data Types

    ax = plt.subplot(111)  # no visible frame
    table(ax, nba)  # where df is your data frame

    plt.savefig('mytable.png')

    print("a")

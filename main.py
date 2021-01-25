import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sn
from pandas import DataFrame


# References:
# 1 - https://realpython.com/pandas-python-explore-dataset/


def create_histogram(array_input, file_name: str) -> bool:
    try:
        _ = plt.hist(array_input)
        plt.title(file_name)
        fig = plt.figure()
        plt.show()
        fig.savefig("histogram_" + file_name + '.png')

    except Exception as ex:
        print(ex)
        return False
    return True


class SoftwareQualityEvaluationExam:
    _dataset: DataFrame
    _dataset_value: numpy.ndarray
    _matrix: numpy.ndarray

    def __init__(self, dataset: DataFrame):
        self._dataset: DataFrame = dataset
        # Dal dataset si prendono solo valori:
        # togliere le prime 3 colonne (name,version,name) + la prima riga (intersazione)
        # nota - > la prima riga di instestazione è già rimossa da panda ..
        # poi conveto tutti i valori in float valori
        # https://stackoverflow.com/questions/44965192/slicing-columns-in-python
        temp = self._dataset.values[:, 3:]
        self._dataset_value = temp.astype(np.float)

        # nuova matrice con solo le variabili effettivamente da analizzare
        # https://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
        self._matrix = self._dataset_value[:, [0, 3, 5, 10, 20]]
        print("MATRIX")
        print(self._matrix)

    def run(self):
        # - Grandezza mia matrice
        size = self._matrix.shape
        print("count row:" + str(size[0]))
        print("count columns:" + str(size[1]))

        # - Displaying Data Types (del dataset)
        self._dataset.info()

        # - Displaying Data Types
        self._dataset.describe()

        # - Display Covariance Matrix
        # self.__get_covariance_matrix()

        self.show_histograms()

        print("a")

    def show_histograms(self):
        wcm = self._matrix[:, [0]]
        create_histogram(wcm, "wcm")

        cbo = self._matrix[:, [1]]
        create_histogram(cbo, "cbo")

        lcom = self._matrix[:, [2]]
        create_histogram(lcom, "lcom")

        loc = self._matrix[:, [3]]
        create_histogram(loc, "loc")

        bug = self._matrix[:, [4]]
        create_histogram(bug, "bug")

    def __get_covariance_matrix(self):
        # https://datatofish.com/covariance-matrix-python/
        covMatrix = np.cov(self._matrix, bias=True)
        print(covMatrix)
        sn.heatmap(covMatrix, annot=True, fmt='g')
        fig = plt.figure()
        plt.show()
        fig.savefig('covariance_matrix.png')


def save_image_example():
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    x = np.arange(10)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label='$y = numbers')
    plt.title('Legend inside')
    ax.legend()
    plt.show()
    fig.savefig('plot.png')


def save_table():
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()
    plt.show()


def get_dataset_path():
    directory = pathlib.Path().absolute()
    path = os.path.join(directory, 'dataset.csv')
    return path


if __name__ == '__main__':

    try:
        dataset_path = get_dataset_path()
        nba = pd.read_csv(dataset_path)
        exam = SoftwareQualityEvaluationExam(nba)
        exam.run()

    except Exception as e:
        print(e)

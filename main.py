import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sn
from pandas import DataFrame
import plotly.graph_objects as go


# References:
# 1 - https://realpython.com/pandas-python-explore-dataset/


def create_histogram(array_input, file_name: str) -> bool:
    try:
        _ = plt.hist(array_input)
        plt.title(file_name)
        fig = plt.figure()
        fig.savefig("histogram_" + file_name + '.png')
        plt.show()

    except Exception as ex:
        print(ex)
        return False
    return True


def create_pie(array_input, file_name: str) -> bool:
    try:

        labels = []
        values = []

        for i in range(0, len(array_input)):
            value: int = int(array_input[i])
            # l'elemento è presente, aumento il suo contatore
            if value in labels:
                index = labels.index(value)
                values[index] = values[index] + 1
            # l'elemento non è
            # lo aggiungo alla lista delle label e segno che ce nè uno
            else:
                values.append(1)
                labels.append(value)

        check_sum = sum(values)

        if len(labels) > 20:
            # todo raggruppare
            a = 1

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax1 = plt.subplots()
        ax1.pie(values, labels=labels)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

        # https://plotly.com/python/pie-charts/
        # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        # fig.show()

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
        # poi conveto tutti i valori in int [non ci sono numeri con virgola]
        # https://stackoverflow.com/questions/44965192/slicing-columns-in-python
        temp = self._dataset.values[:, 3:]
        self._dataset_value = temp.astype(np.int)

        # nuova matrice con solo le variabili effettivamente da analizzare
        # https://stackoverflow.com/questions/8386675/extracting-specific-columns-in-numpy-array
        self._matrix = self._dataset_value[:, [0, 3, 5, 10, 20]]
        print("MATRIX")
        print(self._matrix)

        # Mi salvo i valori in colonna
        self._wcm = self._matrix[:, 0]
        self._cbo = self._matrix[:, 1]
        self._lcom = self._matrix[:, 2]
        self._loc = self._matrix[:, 3]
        self._bug = self._matrix[:, 4]

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

        # self.show_histograms()

        self.show_pie_char()

        print("a")

    def show_histograms(self):
        create_histogram(self._wcm, "wcm")
        create_histogram(self._cbo, "cbo")
        create_histogram(self._lcom, "lcom")
        create_histogram(self._loc, "loc")
        create_histogram(self._bug, "bug")

    def show_pie_char(self):
        create_pie(self._wcm, "wcm")
        create_pie(self._cbo, "cbo")
        create_pie(self._lcom, "lcom")
        create_pie(self._loc, "loc")
        create_pie(self._bug, "bug")

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

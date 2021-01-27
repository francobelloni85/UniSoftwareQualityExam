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
# https://towardsdatascience.com/data-grouping-in-python-d64f1203f8d3
# http://www.sthda.com/english/articles/32-r-graphics-essentials/132-plot-grouped-data-box-plot-bar-plot-and-more/
# https://towardsdatascience.com/chi-square-test-for-independence-in-python-with-examples-from-the-ibm-hr-analytics-dataset-97b9ec9bb80a


import pathlib

import os
import matplotlib.pyplot as plt
import pandas as pd

# References:
# 1 - https://realpython.com/pandas-python-explore-dataset/
from sklearn.decomposition import PCA


def load_nba():
    nba = pd.read_csv(get_dataset_path())
    c = []
    for n in nba.columns:
        if n == "name" or n == "version" or n == "name.1":
            continue
        c.append(n)
    return nba[c]


def do_pca(df, n=2, plot_=True):
    pca = PCA(n)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(df)
    print(df.shape)
    print(projected.shape)
    print(pca.explained_variance_ratio_)
    if plot_:
        plt.scatter(projected[:, 0], projected[:, 1],
                    c=df.bug, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('Spectral', 21))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.show()
    print(abs(pca.components_))


def calculate_correlations(df, top=4):
    """
    from https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

    :param df:
    :return:
    """
    print("Data Frame")
    print(df)
    print()

    print("Correlation Matrix")
    print(df.corr())
    print()

    def get_redundant_pairs(df):
        """Get diagonal and lower triangular pairs of correlation matrix"""
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    def get_top_bug_correleations(df, n=top):
        return df.corr().bug.abs().drop(
            labels=["bug"]
        ).sort_values(ascending=False)[:n]

    print("Top Absolute Correlations")
    print(get_top_abs_correlations(df, 5))
    print()

    print("bug-Correlation Vector")
    bc = get_top_bug_correleations(df, top)
    print(bc)
    print()
    return bc


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


def crete_scatterplots(array_input_A, array_input_B, array_input_A_name, array_input_B_name):
    # https://pythonspot.com/matplotlib-scatterplot/

    # Plot
    plt.scatter(array_input_A, array_input_B, alpha=0.5)
    plt.title('Scatter plot: ' + array_input_A_name + "-" + array_input_B_name)
    plt.xlabel(array_input_A_name)
    plt.ylabel(array_input_B_name)
    plt.show()


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

        features_list = ['wmc', 'cbo', 'lcom', 'loc', 'bug']
        drop_list = []
        for current_features in dataset.values:
            if not current_features in features_list:
                drop_list.append(current_features)

        dataset = dataset.drop(columns=drop_list)
        print(dataset.shape)

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

        # self.show_pie_char()

        # self.show_scatter_plots()

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

    def show_scatter_plots(self):
        # Bug
        crete_scatterplots(self._bug, self._wcm, "bug", "wcm")
        crete_scatterplots(self._bug, self._cbo, "bug", "cbo")
        crete_scatterplots(self._bug, self._lcom, "bug", "lcom")
        crete_scatterplots(self._bug, self._loc, "bug", "loc")

        # wcm
        crete_scatterplots(self._wcm, self._bug, "wcm", "bug")
        crete_scatterplots(self._wcm, self._cbo, "wcm", "cbo")
        crete_scatterplots(self._wcm, self._lcom, "wcm", "lcom")
        crete_scatterplots(self._wcm, self._loc, "wcm", "loc")

        # cbo
        crete_scatterplots(self._cbo, self._bug, "cbo", "bug")
        crete_scatterplots(self._cbo, self._wcm, "cbo", "wcm")
        crete_scatterplots(self._cbo, self._lcom, "cbo", "lcom")
        crete_scatterplots(self._cbo, self._loc, "cbo", "loc")

        # lcom
        crete_scatterplots(self._lcom, self._bug, "lcom", "bug")
        crete_scatterplots(self._lcom, self._wcm, "lcom", "wcm")
        crete_scatterplots(self._lcom, self._cbo, "lcom", "cbo")
        crete_scatterplots(self._lcom, self._loc, "lcom", "loc")

        # _loc
        crete_scatterplots(self._loc, self._bug, "loc", "bug")
        crete_scatterplots(self._loc, self._wcm, "loc", "wcm")
        crete_scatterplots(self._loc, self._cbo, "loc", "cbo")
        crete_scatterplots(self._loc, self._lcom, "loc", "lcom")

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
        my_frame = load_nba()
        # pca is useless, we want maximum correlation with a column!!!
        # do_pca(df, 4, False)
        bug_correlations = calculate_correlations(my_frame)

        print(bug_correlations)
        print()

        dataset_path = get_dataset_path()
        nba = pd.read_csv(dataset_path)
        exam = SoftwareQualityEvaluationExam(nba)
        exam.run()

    except Exception as e:
        print(e)

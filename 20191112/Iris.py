import pydot
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target

    # Visualize iris dataset
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

    clf = DecisionTreeClassifier(max_depth=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Test accuracy: %.4f' % np.mean(y_pred == y_test))

    dot_data = export_graphviz(clf, feature_names=['sepal length', 'sepal width'])
    (graph, ) = pydot.graph_from_dot_data(dot_data)

    file_path = './iris_graph.png' # define your file path for example.., C:/Users/김준호/바탕화면/graph.png
    graph.write_png(file_path)
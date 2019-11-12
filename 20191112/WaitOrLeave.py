import pydot
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz


if __name__ == '__main__':

    df = pd.read_csv('./WaitOrLeave.csv', delimiter=',')

    header = list(df.columns)[1:]
    data = df.values
    X, y = data[:, 1:], data[:, 0]
    
    # Data Encoding
    encoder = LabelEncoder()
    for i in range(X.shape[1]):
        X[:, i] = encoder.fit_transform(X[:, i])
    
    # Declare DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=2)

    # Training the data
    clf.fit(X, y)

    # create dot data for visualization
    dot_data = export_graphviz(clf, feature_names=header)
    (graph, ) = pydot.graph_from_dot_data(dot_data)
    
    file_path = './wait_or_leave_graph.png' # define your file path for example.., C:/Users/김준호/바탕화면/graph.png
    graph.write_png(file_path)


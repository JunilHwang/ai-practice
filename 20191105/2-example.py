import pydot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

  df = pd.read_csv('auto-mpg.csv', delimiter=',')
  data = df.values

  encoder = LabelEncoder()
  # car_name encode
  data[:, -1] = encoder.fit_transform(data[:, -1])
  # sort by mpg column
  data = data[data[:, 0].argsort()]

  X, y = data[:, 1:], data[:, 0]

  for class_num, i in enumerate(range(0, y.shape[0], 150)):
    y[i:i+150] = class_num
  y = y.astype(np.int)
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  clf = DecisionTreeClassifier(max_depth=3)

  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print('Test accuracy: %.3f' % np.mean(y_pred == y_test))
  # graph visualization
  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('graph2.png')

import pydot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__' :

  df = pd.read_csv('auto-mpg.csv', delimiter=',')
  data = df.values
  encoder = LabelEncoder()

  # car name encode
  data[:, -1] = encoder.fit_transform(data[:, -1])

  # sort by mpg column
  data = data[data[:, 0].argsort()]

  x, y = data[:, 1:], data[:, 0]


  for class_num, i in enumerate(range(0, y.shape[0], 150)) :
    y[i:i+150] = class_num
  y = y.astype(np.int)
  
  xTrain, xTest, yTrain, yTest = train_test_split(x, y)
  clf = DecisionTreeClassifier(max_depth = 3) # 3, 5, 7, ...
  clf.fit(xTrain, yTrain)

  yPred = clf.predict(xTest)
  print('Test data accuracy: %.3f' % np.mean(yPred == yTest))

  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('graph.png')
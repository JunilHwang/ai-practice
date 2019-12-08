import pydot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sys import argv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__' :

  depth = int(argv[1]) if len(argv) > 1 else None
  min_split = int(argv[2]) if len(argv) > 2 else 2
  min_leaf = int(argv[3]) if len(argv) > 3 else 1

  df = pd.read_csv('dstest.txt', delimiter=',')
  data = df.values;
  x = data[:, 1:]
  y = (data[:, 0] == 1)
  
  xTrain, xTest, yTrain, yTest = train_test_split(x, y)
  clf = DecisionTreeClassifier(max_depth = depth, min_samples_split = min_split, min_samples_leaf = min_leaf)
  clf.fit(xTest, yTest)

  yPred = clf.predict(xTest)
  acc = np.mean(yPred == yTest)
  print('Test data accuracy: %.3f' % acc)

  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('graph_%s_%s_%s_%.3f.png' % (depth, min_split, min_leaf, acc))

  exit()

  testClass1 = xTest[yTest == 0]
  plt.scatter(testClass1[:, 0], testClass1[:, 1], label = 'testClass1')

  testClass2 = xTest[yTest == 1]
  plt.scatter(testClass2[:, 0], testClass2[:, 1], label = 'testClass2')

  # trainClass1 = xTrain[yTrain == 0]
  # plt.scatter(trainClass1[:, 0], trainClass1[:, 1], label = 'trainClass1')

  # trainClass2 = xTrain[yTrain == 1]
  # plt.scatter(trainClass2[:, 0], trainClass2[:, 1], label = 'trainClass2')

  plt.legend(loc='best')
  plt.show()
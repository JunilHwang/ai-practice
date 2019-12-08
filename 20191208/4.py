import pydot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sys import argv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__' :


  df = pd.read_csv('table1.txt', delimiter=',', header=None)
  data = df.values;
  x = data[:, 0:3]
  y = (data[:, -1] == 1)

  
  xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1, shuffle = False)
  clf = DecisionTreeClassifier()
  clf.fit(xTrain, yTrain)

  yPred = clf.predict(xTrain)
  acc = np.mean(yPred == yTrain)
  print('Test data accuracy: %.3f' % acc)

  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('problem4.png')

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
import pydot
import numpy as np
# print(np.matrix([ [ 1, 2 ], [ 2, 3 ]]))
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__' :
  np.random.seed(0)
  x = np.random.rand(300, 2)
  y = (x[:,0] > 0.3) & (x[:,0] < 0.7) & (x[:, 1] > 0.3) & (x[:, 1] < 0.7)

  mask = np.random.permutation(len(x))[:5]
  y[mask] = ~y[mask]

  # decision tree 실습에 사용할 데이터 셋
  
  xTrain, xTest, yTrain, yTest = train_test_split(x, y)
  clf = DecisionTreeClassifier(max_depth = 3) # 3, 5, 7, ...
  clf.fit(xTrain, yTrain)

  yPred = clf.predict(xTest)
  print('Test data accuracy: %.3f' % np.mean(yPred == yTest))

  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('graph.png')

  # xClass1 = x[y == 0]
  # plt.scatter(xClass1[:, 0], xClass1[:, 1], color = (1.0, 0, 0), label = 'class1')

  # xClass2 = x[y == 1]
  # plt.scatter(xClass2[:, 0], xClass2[:, 1], color = (0, 0, 1.0), label = 'class2')

  # plt.legend(loc='best')
  # plt.show()
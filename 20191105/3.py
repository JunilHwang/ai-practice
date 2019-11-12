import pydot
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

  np.random.seed(0)

  X = np.random.rand(300, 2)
  y = (X[:,0] > 0.3) & (X[:,0] < 0.7) & (X[:,1] > 0.3) & (X[:,1] < 0.7)

  # randomly filps some labels
  mask = np.random.permutation(len(X))[:5]
  y[mask] = ~y[mask]

  X_train, X_test, y_train, y_test = train_test_split(X, y)
  clf = DecisionTreeClassifier(max_depth=3)

  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print('Test accuracy: %.3f' % np.mean(y_pred == y_test))
    # graph visualization
  clf_dot = export_graphviz(clf)
  (graph, ) = pydot.graph_from_dot_data(clf_dot)
  graph.write_png('graph.png')
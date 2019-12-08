import pydot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sys import argv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__' :

  depth = int(argv[1]) if len(argv) > 1 else None
  min_split = int(argv[2]) if len(argv) > 2 else 2
  min_leaf = int(argv[3]) if len(argv) > 3 else 1

  df = pd.read_csv('dstest.txt', delimiter=',')
  data = df.values;
  x = data[:, 1:]
  y = (data[:, 0] == 1)
  
  xTrain, xTest, yTrain, yTest = train_test_split(x, y)

  tree = DecisionTreeClassifier(max_depth=2)
  tree.fit(xTrain, yTrain)
  yTrainPred = tree.predict(xTrain)
  yTestPred = tree.predict(xTest)

  treeTrain = accuracy_score(yTrain, yTrainPred)
  treeTest = accuracy_score(yTest, yTestPred)
  print('Decision tree train/test accuracies %.3f/%.3f' % (treeTrain, treeTest))

  for nEstimators in [10, 15, 20, 25, 30, 35, 40, 45, 50] :

    RF = RandomForestClassifier(max_depth=2, n_estimators=nEstimators)
    RF.fit(xTrain, yTrain)
    yTrainPred = RF.predict(xTrain)
    yTestPred = RF.predict(xTest)

    RFTrain = accuracy_score(yTrain, yTrainPred)
    RFTest = accuracy_score(yTest, yTestPred)
    print('<N: %d> train/test %.3f/%.3f' % (nEstimators, RFTrain, RFTest))
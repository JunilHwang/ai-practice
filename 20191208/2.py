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

  for n_increment in [10, 15, 20, 25, 30, 35, 40, 45, 50] :

    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=n_increment, learning_rate=0.1, random_state=0)
    ada.fit(xTrain, yTrain)
    yTrainPred = ada.predict(xTrain)
    yTestPred = ada.predict(xTest)

    adaTrain = accuracy_score(yTrain, yTrainPred)
    adaTest = accuracy_score(yTest, yTestPred)
    print('AdaBoost train/test accuracies %.3f/%.3f' % (adaTrain, adaTest))
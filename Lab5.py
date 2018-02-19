__author__ = 'Trenton'
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np
from sklearn import preprocessing


#Create some type of data structure to hold a node (a.k.a. neuron).
class Neuron:
    #Store the set of input weights for each node.
    InputWeights = []
    numNodes = 0
    bias = -1.00 #Account for a bias input.
    biasWeight = random.randint(-99,99)/100.0
    Threshold = 0

#Provide a way to create a layer of nodes of any number (this should be easily specified via a parameter).
    def __init__(self, columns):
        self.numNodes = len(columns)
        #inishalize the weights
        for w in columns:
            self.InputWeights.append((random.randint(-99,99))/100.0)
        #print(self.InputWeights) - just for testing



    #Be able to take input from a dataset instance (with an arbitrary number of attributes) and have each node produce an output (i.e., 0 or 1) according to its weights.
    def produceOutput(self, x_instance):
        z = 0
        for x , y in zip(x_instance, self.InputWeights):
            z += x * y
        z += self.bias * self.biasWeight

        if z > 0:
            return 1
        else:
            return 0







def main():
    #Be able to load and process at least the following two datasets:
    ####Iris (You didn't think we could leave this out did you!)
    #You should appropriately normalize each data set.
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    IrisN = Neuron(X_test[0])
    print("iris")
    for x in X_train:
        print(IrisN.produceOutput(x))

    ####Pima Indian Diabetes
    #You should appropriately normalize each data set.
    pima = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
    pima.columns = ["numPreg", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPF", "age", "class"]
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_accepted:
        pima[column] = pima[column].replace(0, np.NaN)
        mean = int(pima[column].mean(skipna=True))
        pima[column] = pima[column].replace(np.NaN, mean)
    pimaNP = pima.as_matrix()
    pima_targets = []
    for x in pimaNP:
        pima_targets.append(x[8])
    pimaNP = np.delete(pimaNP, 8, 1)
    targetNP = np.array(pima_targets)
    # X_train, X_test, y_train, y_test = train_test_split(VONP, VOTargets, test_size=0.3)
    X_train, X_test, y_train, y_test   = train_test_split(pimaNP, targetNP, test_size=.3)
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    PimaN = Neuron(X_test[0])
    print("pima")
    for x in X_train:
        print(IrisN.produceOutput(x))







if __name__ == "__main__":
    main()
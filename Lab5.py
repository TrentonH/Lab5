__author__ = 'Trenton'
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Create some type of data structure to hold a node (a.k.a. neuron).
class Neuron:
    #Store the set of input weights for each node.
    InputWeights = []
    numNodes = 0
    bias = -1.00 #Account for a bias input.
    biasWeight = random.randint(-99,99)/100.00
    Activation = 0.00
    Error = 0

#Provide a way to create a layer of nodes of any number (this should be easily specified via a parameter).
    def __init__(self, columns):
        self.numNodes = len(columns)
        #inishalize the weights
        for w in columns:
            self.InputWeights.append((random.randint(-99,99))/100.00)
        #print(self.InputWeights) - just for testing



    #Be able to take input from a dataset instance (with an arbitrary number of attributes) and have each node produce an output (i.e., 0 or 1) according to its weights.
    def produceOutput(self, x_instance):
        z = 0.00
        for x , y in zip(x_instance, self.InputWeights):
            z += x * y
        z += self.bias * self.biasWeight

        z = z * -1 #set up the -z for the equation
        self.Activation = 1 / (1 + math.e**z)#calculates the activation
        return self.Activation
    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

def feedForward(Nurons, imputs):
    outputs = []
    num_of_nurons = len(Nurons)
    for x in range (0,num_of_nurons):
        outputs.append(Nurons[x].produceOutput(imputs))
    return outputs

def calcAcc(list1, list2):
    length = len(list1)
    if length < len(list2):
            length = len(list2)
    counter = 0
    for x in range (0,length):
        if list1[x] == list2[x]:
            counter += 1
    counter = counter / length
    return counter

def printAcc(accuracyList):
    x = []
    y = []
    for z in range(0, 100):
        y.append(z/100.00)
    for z in range(0, len(accuracyList)):
        x.append(z)
    plt.plot(x,)
    plt.show()



def main():
    #Be able to load and process at least the following two datasets:
    ####Iris (You didn't think we could leave this out did you!)
    #You should appropriately normalize each data set.
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)
    #IrisN = Neuron(X_test[0])
    print("iris {}".format(X_train[0]))
    answers = []
    #for x in X_train:
        #print(IrisN.produceOutput(x))
    NeuronList = [[]]
    for w in range (0, len(X_train[0])):
        a = Neuron(X_test[0])
        NeuronList[0].append(a)
    print("input weights 1 section = {}".format(NeuronList[0][0].InputWeights))
    print("input weights 2 section = {}".format(NeuronList[0][1].InputWeights))
    print("input weights 3 section = {}".format(NeuronList[0][2].InputWeights))
    print("input weights 4 section = {}".format(NeuronList[0][3].InputWeights))
    numLayers = int(input("how many layers do you want for the Iris data set"))
    for x in range (1, numLayers):
        numnodes = int(input("how many nodes do you want this layer of the Iris data set"))
        for y in range (0, numnodes):
            NeuronList[x].append(Neuron(X_test[0]))
            print("input weights = {}".format(NeuronList[x][y].InputWeights))
    for a in range (0, len(X_train)):
        z = 0
        NextImput = []
        while NeuronList[z]:
            if z == 0:
                NextImput = feedForward(NeuronList[z], X_train[a])
            else:
                NextImput = feedForward(NeuronList[z], NextImput)
            z += 1
        max = -100
        nodeNum = 0
        print("next input {}".format(NextImput))
        for b in range (0, len(NextImput)):
            if NextImput[b] > max:
                max = NextImput[b]
                nodeNum = b
        print("This is a clasification {}".format(nodeNum))
        answers.append(nodeNum)
    acc = calcAcc(answers,y_train)
    print("acc = {}".format(acc))




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
    #PimaN = Neuron(X_test[0])
    print("pima")
    answers = []
   # for x in X_train:
        #print(PimaN.produceOutput(x))
    NeuronList = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for w in X_train:
        NeuronList[0].append(Neuron(X_test[0]))
    numLayers = int(input("how many layers do you want for the pima data set"))
    for x in range (1, numLayers):
        numnodes = int(input("how many nodes do you want this layer of the pima data set"))
        for y in range (0, numnodes):
            NeuronList[x].append(Neuron(X_test[0]))
    for a in range (0, len(X_train)):
        z = 0
        NextImput = []
        while NeuronList[z]:
            if z == 0:
                NextImput = feedForward(NeuronList[z], X_train[a])
            else:
                NextImput = feedForward(NeuronList[z], NextImput)
            z += 1
        max = 0
        nodeNum = 0
        for b in range (0, len(NextImput)):
            if NextImput[b] > max:
                max = NextImput[b]
                nodeNum = b
        print("This is a clasification {}".format(nodeNum))
        answers.append(nodeNum)
    calcAcc(answers,y_train)
    print("acc = {}".format(acc))






if __name__ == "__main__":
    main()
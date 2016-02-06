# JMJ

# Quick readme. to run this run python pybrain_nn1.py , assuming you have train.csv and test.csv in same file
# . first it reads data into a pybrain Classification Data Set, used for classification algos
# The dataset has only 1 output but we use a Pybrain method convertOneToMany to map a single output into 10 possible
# classifications
# Then build a feed forward neural network with 784 inputs, 10 outputs and 1 hidden layer with 250? neurons
# this is trained 20 times
import csv
import pickle
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import FeedForwardNetwork

from pybrain.supervised.trainers import BackpropTrainer
from numpy import *
from pybrain.structure.modules import SoftmaxLayer, LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.utilities import percentError

X = []
Y = []


def split_test_train(dataset):
    test_data, trndata = dataset.splitWithProportion(0.25)
    test_data.__class__ = ClassificationDataSet
    trndata.__class__ = ClassificationDataSet
    test_data._convertToOneOfMany(bounds=[0, 1])
    trndata._convertToOneOfMany(bounds=[0, 1])
    print "Length of test_data " + str(test_data.__len__())
    print "Length of train data " + str(trndata.__len__())
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    return test_data, trndata


def get_dataset():
    reader = csv.reader(open('train.csv', 'rb'))

    reader.next()  # read header
    # create a dataset, with 784 inputs 1 output
    dataset = ClassificationDataSet(784, 1, nb_classes=10,
                                    class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    for row in reader:
        inputs = map(int, row[1:])
        output = map(int, row[0:1])
        X.append(inputs)
        Y.append(output)
        dataset.addSample(inputs, output)
    # this maps the single target variable into 10 output classes
    dataset._convertToOneOfMany(bounds=[0, 1])
    return dataset


def run_against_testset(net):
    print "Predicting with the neural network"
    test_reader = csv.reader(open('test.csv', 'rb'))

    header = test_reader.next()

    # write output
    predictions_file = open("output1.csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId", "Label"])

    predX = []
    ids = []
    count = 1
    for row in test_reader:
        inputs = map(int, row[0:])
        prediction = net.activate(inputs)
        predicted_answer = argmax(prediction)
        predX.append(predicted_answer)
        ids.append(count)
        count += 1

    open_file_object.writerows(zip(ids, predX))
    predictions_file.close()
    print 'Done.'


def build_feed_forward():
    n = FeedForwardNetwork()
    inlayer = LinearLayer(784)
    hiddenlayer = SigmoidLayer(397)  # mean of 784 & 10
    outputlayer = SoftmaxLayer(10)
    n.addInputModule(inlayer)
    n.addModule(hiddenlayer)
    n.addOutputModule(outputlayer)
    in_to_hidden = FullConnection(inlayer, hiddenlayer)
    hidden_to_out = FullConnection(hiddenlayer, outputlayer)
    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)
    n.sortModules()
    return n


def attempt2(dataset):
    print "Attempt 2"
    fnn = build_feed_forward()
    train_nn(fnn, dataset)
    fileObject = open('neural_network', 'w')
    pickle.dump(fnn, fileObject)
    fileObject.close()
    run_against_testset(fnn)


def train_nn(nn, dataset):
    print "Start training"
    trainer = BackpropTrainer(nn, dataset=dataset, momentum=0.1, verbose=True, weightdecay=0.01)
    trainer.trainUntilConvergence(dataset=dataset, maxEpochs=120, verbose=True, continueEpochs=10,validationProportion=.25)

    return trainer


def main():
    print "JMJ"
    data_set = get_dataset()
    attempt2(data_set)


if __name__ == '__main__':
    main()

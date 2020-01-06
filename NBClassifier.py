import pandas as pd
import numpy as np
import tkinter.messagebox as msg


# A class that holds all the required data for training a Naive Bayes model
class NaiveBayes:
    # constructor. gets the training dataframe, number of bins and the attributes type dictionary
    def __init__(self, data, binsNumber, attributesDict):
        """

        :type attributesDict: dict
        :type data: DataFrame
        """
        self.atrributesTypeDic = dict()  # initate new attribute to type dictionary, the other one will serve for attribute to possible values
        for att in attributesDict.keys():
            if isinstance(attributesDict[att], list):
                self.atrributesTypeDic[att] = "NOMINAL"
            else:
                self.atrributesTypeDic[att] = "NUMERIC"
        self.binsNumber = binsNumber
        self.attributesValuesDict = attributesDict
        self.attBins = dict()  # attributes to calculated cuts dictionary, to discretize test set.
        self.discretize(data)  # discretize and fillna to train set.
        self.trainSet = data
        self.classAttributesProbabilities = dict()
        self.classOccurences = dict()  # class-> number_of_occurences dictionary
        for cls in attributesDict['class']:  # count class occurences
            self.classOccurences[cls] = self.trainSet.loc[self.trainSet['class'] == cls].shape[0]

    # complete missing values and discretize numeric attributes
    def discretize(self, frame):
        try:
            # fillna for nominal attributes
            for att in filter(lambda x: self.atrributesTypeDic[x] == "NOMINAL", self.atrributesTypeDic.keys()):
                frame[att].fillna(frame[att].mode()[0], inplace=True)
            # fillna and discretize numeric attributes
            for att in filter(lambda x: self.atrributesTypeDic[x] == "NUMERIC", self.atrributesTypeDic.keys()):
                col = frame[att]
                if (sum(col.isnull()) != 0):  # if there are missing values, fill them with mean values
                    col.fillna(col.mean(), inplace=True)
                if isinstance(self.attributesValuesDict[att],
                              list):  # if the attribute was already discretized, use calculated bins. (relevant onlt for test set)
                    col = pd.cut(col, self.attBins[att], labels=self.attributesValuesDict[att], include_lowest=True)
                else:
                    bins = pd.cut(col, self.binsNumber, labels=range(self.binsNumber), include_lowest=True,
                                  retbins=True)
                    col = bins[0]
                    self.attBins[att] = bins[1]  # save bins
                    self.attributesValuesDict[att] = range(self.binsNumber)
                frame[att] = col
        except Exception as e:
            print(e)

    # returns arrgmax of probability of being in each class
    def getClassification(self, dictionary):
        dictionary = dict(dictionary)
        maxx = 0
        label = ""
        for cls in dictionary.keys():
            if dictionary[cls] > maxx:
                maxx = dictionary[cls]
                label = cls
        return label

    # classify each record in the test set
    def classify(self, testSet):
        if len(self.classAttributesProbabilities.keys()) == 0:
            msg.showerror("Naive Bayes Classifier", "Model not trained")
            return
        results = dict()  # dictionary of results. will be returned
        self.discretize(testSet)  # discretize test set first
        totalRows = self.trainSet.shape[0]
        for i, row in testSet.iterrows():
            classToProbablity = dict()
            for classValue in self.attributesValuesDict['class']:
                sumProbs = 1
                try:  # sum probabilities according to attribute values of given record
                    for attribute in self.attributesValuesDict.keys():
                        if attribute == "class":
                            continue
                        attval = row[attribute]
                        sumProbs *= self.classAttributesProbabilities[classValue][attribute][attval]
                    classToProbablity[classValue] = sumProbs * (self.classOccurences[classValue] / float(totalRows))
                except Exception as e:
                    print(e)
            results[i + 1] = self.getClassification(classToProbablity)
        return results

    # train model according to test set
    def train(self):
        totalRows = self.trainSet.shape[0]
        for cls in self.attributesValuesDict['class']:
            n = self.classOccurences[cls]
            self.classAttributesProbabilities[cls] = dict()  # initiate probabilities dictionary
            for attribute in self.attributesValuesDict.keys():
                if attribute == "class":
                    continue
                self.classAttributesProbabilities[cls][attribute] = dict()
                p = 1.0 / len(self.attributesValuesDict[attribute])
                for attributeValue in self.attributesValuesDict[attribute]:
                    nc = self.trainSet.loc[
                        (self.trainSet['class'] == cls) & (self.trainSet[attribute] == attributeValue)].shape[
                        0]  # number of records with class=cls and attribute=attributeValue
                    m_estimate = (nc + 2 * p) / (n + 2)  # calculate m-estimate
                    self.classAttributesProbabilities[cls][attribute][attributeValue] = m_estimate

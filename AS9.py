"""Binary Classification
First, you should read the NASA data that is given to you as a .csv file ( NasaData.csv ).
Use pandas read_csv function for this.

After reading the data, build binary classification models with KNeighborsClassifier,
from SKLearn. The model gets metrics as feature set and predicts either a defective (1) or not defective (0) label.

Take 75% of data as training set and 25% of it as test set. To eliminate the randomness when
splitting the data, you should run each classification technique 30 times with seeds from [1 to 30].

Using model_selection's train_test_split function, randomly select 1/4 of your dataset as training
and 3/4 as testset. Calculate accuracies per technique and repeat this for a total of 30 random runs
(every run will use a different random seed in train_test_split and return a separate accuracy value per model).

Visualize the distribution of the accuracies for each model in a single box plot, where The X_axis is
the classification technique that you've applied and the Y_axis is the accuracies.

* Note1 you must use pandas for both reading from CSV and visualizing boxplots *"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

scores = list()
nasa = pd.read_csv('NasaData.csv')
x = nasa.iloc[:, 0:-1]
y = nasa.iloc[:, -1]

for i in range(1, 31):
    # the test size is the % of data used to test the model, the rest of the data
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=i)
    # is used to train the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(xtrain, ytrain)
    scores.append(knn.score(xtest, ytest))
scores
plot = pd.DataFrame({"KNN": scores})
axes = plot.boxplot(column=['KNN'], return_type="axes")
# axes.set_ylim([0,1])
# print("Test Score: {:.3f}".format(knn.score(xtest,ytest)))


"""Part B. KNN complexity tuning
In this part, we want to tune the value k in kNN for our NASA dataset. To do so, you should find a sweet
spot that the model is neither overfitted nor underfitted. Here again take the NASA dataset and apply the
model_selection's train_test_split with 75% training and 25% test data, but with a fix random_state=42.
Then build a K-Nearest-Neighbors model using k=1,3,5,..,49. Finally, plot the accuracy of your models on
the training dataset and the testing dataset, using two lines in one plot.
Using this plot identify what the best value is for k."""

nasa = pd.read_csv('NasaData.csv')
x = nasa.iloc[:, 0:-1]
y = nasa.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)
trainacc = list()
testacc = list()
DifferentK = [i*2-1 for i in range(1, 25)]
for k in DifferentK:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    testacc.append(knn.score(xtest, ytest))
    trainacc.append(knn.score(xtrain, ytrain))


def PlotDrawer(Plot_one, Plot_Two, X_axis):
    plt.plot(X_axis, Plot_one, label="training accuracy")
    plt.plot(X_axis, Plot_Two, label="test accuracy")
    plt.ylabel('Mean Accuracy')
    plt.xlabel('K-value')
    plt.legend()
    plt.show()


# right is underfit (train and test acc are close(higher k)) left is overfit (train and test acc are very diff(lower k))
PlotDrawer(trainacc, testacc, DifferentK)
# best value for k is around 15

"""Part C. Regression
In this section, we will use a new data set which is related to the performance of several CPUs.
These CPUs are of different specifications, and you have the estimated relative performance(ERP)
metric per CPU, in this data set.

columns of data set are as follows:

MYCT: machine cycle time in nanoseconds (integer)
MMIN: minimum main memory in kilobytes (integer)
MMAX: maximum main memory in kilobytes (integer)
CACH: cache memory in kilobytes (integer)
CHMIN: minimum channels in units (integer)
CHMAX: maximum channels in units (integer)
PRP: published relative performance (integer)
ERP: estimated relative performance from the original article (integer)

Read the data that is given to you as a CSV file ("CPU_Performance.csv") and take 75%
of it as training set and 25% of it as test set with random_state=42.

Use default KNeighborsRegressor, to predict ERP using the other columns as features.

To see how good you can predict on new CPUs performance, print the score of the model on training set and test set data.

Use the default setup for the models."""

data = pd.read_csv("CPU_Performance.csv")
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)
knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(xtrain, ytrain)
print(knr.score(xtrain, ytrain))
print(knr.score(xtest, ytest))

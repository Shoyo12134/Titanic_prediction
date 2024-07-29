# Titanic_prediction
## Problem description
In this challenge, we ask you to build a predictive model that answers the question:
“what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
## Data
In this competition, you’ll gain access to two similar datasets that include passenger information like name, age, gender, socio-economic class, etc.
One dataset is titled train.csv and the other is titled test.csv.Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly,
will reveal whether they survived or not, also known as the “ground truth”.The test.csv dataset contains similar information
but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.Using the patterns you find in the train.csv data, predict whether the other 
418 passengers on board (found in test.csv) survived.
## Evaluations
the program gets to a accuracy of 95 percent to continue
## Features
* Variable	Definition	Key
* survival	Survival	0 = No, 1 = Yes
* pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
* sex	Sex	
* Age	Age in years	
* sibsp	# of siblings / spouses aboard the Titanic	
* parch	# of parents / children aboard the Titanic	
* ticket	`Ticket number`	
* fare	`Passenger fare`	
* cabin	Cabin number	
* embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

* sibsp: The dataset defines family relations in this way...
* Sibling = brother, sister, stepbrother, stepsister
* Spouse = husband, wife (mistresses and fiancés were ignored)


##Model used is 
Since it is a classification problem as we have to predict weather a person has surived or not i.e-`0` or `1`
we will use technique of classification problem


## dependncy to install
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
## modules to be install
from sklearn  import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

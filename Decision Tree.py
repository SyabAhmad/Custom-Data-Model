
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("customData.csv")
print(data.head(5))

xData = data.iloc[:, :-1]

yData = data.iloc[:,1]

regresser = DecisionTreeRegressor()

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=18, random_state=1)

regresser.fit(xTrain, yTrain)

prediction = regresser.predict(xTest)
accuracy = regresser.score(xTrain, yTrain)
print(accuracy)
print(prediction)
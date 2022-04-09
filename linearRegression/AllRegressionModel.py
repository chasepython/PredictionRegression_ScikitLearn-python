import pandas as pd
import numpy as np
from pyparsing import anyCloseTag      # for hstack()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances_argmin
from sklearn.model_selection import train_test_split    # for spliting data and testing overfitting
from sklearn.model_selection import cross_val_score     # for cross-validation
from sklearn.linear_model import Lasso
import matplotlib.pyplot as pyplot


# simple Regression
"""
analysis_data = pd.read_csv("XXX.csv")

lm = LinearRegression()

y = analysis_data["Y"].values.reshape(-1,1)   # y is dependent variable
x = analysis_data["X"].values.reshape(-1,1)   # x is independent variable

lm.fit(x,y)     # train the model

print("Coefficients:", lm.coef_)
print("Intercept:", lm.intercept)
print("R Square:", lm.score(x, y))   # R^2

"""

# Quadratic Regression
"""
analysis_data = pd.read_csv("XXX.csv")

lm = LinearRegression()

y = analysis_data["Y"].values.reshape(-1,1)   # Y is dependent variable
x = analysis_data["X"].values.reshape(-1,1)
xSquare = pow(x, 2)     # 2 means square
x_hstack = np.stack((x, xSquare))       # x_hastack is a matrix！

lm.fit(x_hstack,y)     # train the model

print("Coefficients:", lm.coef_)
print("Intercept:", lm.intercept)
print("R Square:", lm.score(x_hstack, y))   # R^2

"""


# Multiple Regression & Dummy Variable
"""
analysis_data = pd.read_csv("XXX.csv")

lm = LinearRegression()

# -----------------------------------------
y = analysis_data["Y"].values.reshape(-1,1)   # y is dependent variable

x1 = analysis_data["X1"].values.reshape(-1,1)

x2 = analysis_data["X2"].values.reshape(-1,1)
x2Square = pow(x2, 2)

x3 = analysis_data["X3"].values.reshape(-1,1)

Dummy_x = pd.get_dummies(analysis_data["X_D"])       # X_D is Dummy variable"類別變數"
# -------------------------------------------

All_x = np.hstack((Dummy_x-1,x1,x2,x2Square,x3))      # All_X is a matrix! (X_D-1 or not) is OK, if just for predict! 



lm.fit(All_x, y)     # train the model

print("Coefficients:", lm.coef_)
print("Intercept:", lm.intercept)
print("R Square:", lm.score(All_x, y))   # R^2

"""


# Predict Model & Multiple Regression & Dummy Variable 
"""
analysis_data = pd.read_csv("XXX.csv")      # initial data

analysis_future = pd.read_csv("XXX_future.csv")     # "future" is a section data for training model.

lm = LinearRegression()

# -----------------------------------------
y = analysis_data["Y"].values.reshape(-1,1)   # y is dependent variable

x1 = analysis_future["X1"].values.reshape(-1,1)     # x1 is prediction

x2 = analysis_future["X2"].values.reshape(-1,1)     # x2 and x2Square are prediction
x2Square = pow(x2, 2)

x3 = analysis_future["X3"].values.reshape(-1,1)     # x3 is prediction

Dummy_x = pd.get_dummies(analysis_future["X_D"])       # X_D is Dummy variable"類別變數", x3 is prediction
# -------------------------------------------

All_x_future = np.hstack((Dummy_x-1,x1,x2,x2Square,x3))      # All_x is a matrix! (X_D-1 or not) is OK, if just for predict! 

y_predict = lm.predict(All_x_future)

print(y_predict)

pyplot.plot(analysis_data["X1"].values, analysis_data["Y"].values, "bo")    # initial data for reference
pyplot.plot(x1, y_predict, "ro")
pyplot.show()

"""


# Predict Model & Test overfitting.  
"""
#You nedd to Change initial data of independent variables for analyzing. 
#Dummy variable by Manual spliting.
#Spilit makes R^2 random => solution: cross-validation

analysis_data = pd.read_csv("XXX.csv")      # initial data

analysis_fitting = pd.read_csv("XXX_testfitting.csv")     # Can test _underfitting、_overfitting for sure.

All_x = analysis_fitting.drop(["Y"], axis = 1)   # All_x is all independent variables
y = analysis_fitting["Y"].values.reshape(-1,1)   # y is dependent variable

train_x, valid_x, train_y, valid_y = train_test_split(All_x, y, test_size = 0.%d)   # can set 0.%d ratio, and its result is random spiliting.

lm = LinearRegression()
lm.fit(train_x, train_y)    # use trainning data for regression model bulid.
print("R Square:    ", lm.score(train_x, train_y))  # testing error

predicted_y = lm.predict(valid_x)   # predict Y by valid independent variables.
rss = ((predicted_y - valid_y) ** 2).mean()
tss = ((valid_y.mean() - valid_y) ** 2).mean()
print(1 - rss/tss)  # prediction error. compare "prediciton error" with "testing error". => {Good testing error != Good prediction error}

"""

# Cross-Validation
"""
analysis_data = pd.read_csv("XXX_testfitting.csv")
x = analysis_data(["Y"], axis = 1)
y = analysis_data["Y"]

lm = LinearRegression()
print(cross_val_score(lm, x, y, cv = 4).mean())     # 4-fold cross-vaidation for validation fitting. it's fixed value.

"""


# LASSO regression => solve multiple variables
"""
analysis_Lasso = pd.read_csv("XXX_testfitting.csv")

x = analysis_Lasso(["Y"], axis = 1)
y = analysis_Lasso(["Y"])

lm_normal = LinearRegression()                              # all variables => for comparison "Lasso"
print(cross_val_score(lm_normal, x, y, cv = 4).mean())      # No penalty funtcion => R^2 


#-- alpha can use loop for trial. ex. 0.001~100 --#
for i in np.arange(0.01,100.01,0.01):
    i = round(i,2)                                         # Round number to the nearest hundredth.
    lm_lasso = Lasso(alpha = i, max_iter = 1000000)    
    print(cross_val_score(lm_lasso, x, y, cv = 4).mean())   

"""

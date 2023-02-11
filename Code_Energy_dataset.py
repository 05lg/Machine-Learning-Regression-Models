
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Importing dataset and examining it
dataset = pd.read_csv("Energy_dataset.csv") #Location of the dataset
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
dataset['school_day'] = dataset['school_day'].map({'N': 0,'Y': 1})
dataset['holiday'] = dataset['holiday'].map({'N': 0,'Y': 1})
print(dataset.info())


# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()

# Dividing dataset into label and feature sets
X = dataset.drop(['demand','max_temperature','frac_at_neg_RRP','RRP_positive'], axis = 1) # Features
Y = dataset['demand'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

print(X.info())

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Linear Regression without Regularization
# Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter' using Grid Search
sgdr = SGDRegressor(random_state = 1, penalty = None)
grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[10000, 20000, 30000, 40000]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("r2: ", best_result)

Adj_r2 = 1-(1-best_result)*(1682-1)/(1682-9-1)
print("Adjusted r2: ", Adj_r2)

'''
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

where, n = number of observations in training data, p = number of features
'''

best_model = gd_sr.best_estimator_
print("Intercept: ", best_model.intercept_)

print(pd.DataFrame(zip(X.columns, best_model.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))

# Linear Regression with Regularization
# Tuning the SGDRegressor parameters 'eta0' (learning rate) and 'max_iter', along with the regularization parameter alpha using Grid Search
sgdr = SGDRegressor(random_state = 1, penalty = 'elasticnet')
grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter':[1000, 5000, 10000, 20000, 30000, 40000],'alpha': [.001, .01, .1, 1,10, 100], 'l1_ratio': [0,0.25,0.5,0.75,1]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("r2: ", best_result)

Adj_r2 = 1-(1-best_result)*(1682-1)/(1682-9-1)
print("Adjusted r2: ", Adj_r2)

'''
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

where, n = number of observations in training data, p = number of features
'''

best_model = gd_sr.best_estimator_
print("Intercept: ", best_model.intercept_)

print(pd.DataFrame(zip(X.columns, best_model.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))

##################################################################################
# Implementing Random Forest Regression
# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfr = RandomForestRegressor(criterion='squared_error', max_features='sqrt', random_state=1)
grid_param = {'n_estimators': [10,20,30,40,50,100]}

gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("r2: ", best_result)

Adj_r2 = 1-(1-best_result)*(1682-1)/(1682-9-1)
print("Adjusted r2: ", Adj_r2)

'''
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

where, n = number of observations in training data, p = number of features
'''

featimp = pd.Series(gd_sr.best_estimator_.feature_importances_, index=list(X)).sort_values(ascending=False) # Getting feature importances list for the best model
print(featimp)

# Selecting features with higher sifnificance and redefining feature set
X_ = dataset[['demand_pos_RRP','RRP','solar_exposure','min_temperature','demand_neg_RRP']]

feature_scaler = StandardScaler()
X_scaled_ = feature_scaler.fit_transform(X_)

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfr = RandomForestRegressor(criterion='squared_error', max_features='sqrt', random_state=1)
grid_param = {'n_estimators': [50,100,150,200,250]}

gd_sr = GridSearchCV(estimator=rfr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled_, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("r2: ", best_result)

Adj_r2 = 1-(1-best_result)*(1682-1)/(1682-9-1)
print("Adjusted r2: ", Adj_r2)

####################################################################################
# Implementing Support Vector Regression
# Tuning the SVR parameters 'kernel', 'C', 'epsilon' and implementing cross-validation using Grid Search
svr = SVR()
grid_param = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [100,1000,10000,100000], 'epsilon': [100,1000,10000]}

gd_sr = GridSearchCV(estimator=svr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("r2: ", best_result)

Adj_r2 = 1-(1-best_result)*(1682-1)/(1682-9-1)
print("Adjusted r2: ", Adj_r2)

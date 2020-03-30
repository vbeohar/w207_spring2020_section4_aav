import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn import linear_model
from scipy import stats
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
np.random.seed(0)

# load csv
df = pd.read_csv('C:\\Users\\Andrew\\Desktop\\Berkeley Courses\\W207 - Applied Machine Learning\\Assignments\\Group Project\\train_cleaned.csv')
# df = df.loc[df['city'] == 'NYC']

# create dummies for some variables that should have them
df2 = pd.get_dummies(data =df, columns = ['city', 'room_type', 'property_type'])

# grab the new dummies
df2 = df2.iloc[:, 43::]

# concat into CSV DF
df = pd.concat([df, df2], axis = 1)

# remove string vars
df = df.select_dtypes(exclude=[np.object])

# drop na
df = df.dropna(axis = 0)

# prepare X, y
y = df['log_price'].values
X = df.drop(['log_price'], axis = 1).values

# shuffle data
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

# split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state = 42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, random_state = 42)

# scale data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# scorer
mse = make_scorer(metrics.mean_squared_error)

# functions
# confidence interval
def confidence_interval(confidence_level, predicted_y, y_test):
    confidence = confidence_level
    squared_errors = (predicted_y - y_test) ** 2
    return np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc = squared_errors.mean(),
                             scale = stats.sem(squared_errors)))

# cross val scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", round(scores.mean(), 3))
    print("Standard deviation:", round(scores.std(), 3))


# result storage
model_name = []
RMSE = []
CI = []
cross_val_mean = []

################## Base Model: Linear Regression ##################
# fit a linear regression model
lm = LinearRegression() # instantiate lm
lm.fit(X_train_full, y_train_full); # fit lm
r2_lm = lm.score(X_train_full, y_train_full) # x, y

# show r2 of regression model
print(round(r2_lm, 3), 'is the r-squared for the regression model')

# generate predictions
predicted_y = lm.predict(X_test) # prediction

# metrics
lm_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(lm_RMSE, 3))

# confidence interval for RMSE
lm_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(lm_ci, 3))

# cross-val on dev data
scores = cross_val_score(lm, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

display_scores(rmse_scores)
# dev data contains some very messed up observations,
# or lm is just struggling to predict,
# as evidenced by cross-val

display_scores(rmse_scores)

# append results
model_name.append('linear model')
RMSE.append(lm_RMSE)
CI.append(lm_ci)
cross_val_mean.append(rmse_scores.mean())

################## Linear Ridge Regression ##################
ridge = Ridge(alpha = 1) # instantiate lm
ridge.fit(X_train, y_train); # fit lm
r2_ridge = ridge.score(X_train, y_train) # r2

# show r2 of regression modela
print(round(r2_ridge, 3), 'is the r-squared for the regression model')

# setup grid search params
params = [{'alpha' : np.random.uniform(100, 500, 100)}]

# grid search hyperparameter alpha
rnd_search_cv = RandomizedSearchCV(ridge, params, n_iter = 30, verbose = 2, cv = 3, scoring = mse, random_state = 42)

# validate search on dev data
rnd_search_cv.fit(X_dev, y_dev) # search over the dev data

rnd_search_cv.best_estimator_ # estimator that was chosen by the search
rnd_search_cv.best_score_ # mean cross-validated score of the best_estimator

# fit again on full data now with best estimator
final_model = rnd_search_cv.best_estimator_

final_model.fit(X_train_full, y_train_full); # fit lm on full data
r2_ridge = ridge.score(X_train_full, y_train_full) # r2

# show r2 of regression model
print(round(r2_ridge, 3), 'is the r-squared for the regression model')

# generate predictions
predicted_y = ridge.predict(X_test) # prediction

# metrics
ridge_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(ridge_RMSE, 3))

# confidence interval for RMSE
ridge_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(ridge_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(final_model, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

display_scores(rmse_scores)

# append results
model_name.append('ridge')
RMSE.append(ridge_RMSE)
CI.append(ridge_ci)
cross_val_mean.append(rmse_scores.mean())

################## Linear Lasso Regression ##################
lasso = linear_model.Lasso(alpha = 0.001)
lasso.fit(X_train, y_train); # fit lm
r2_lasso = lasso.score(X_train, y_train) # x, y

# show r2 of regression model
print(round(r2_lasso, 3), 'is the r-squared for the regression model')

# setup grid search params
params = [{'alpha' : np.random.uniform(0.1, 500, 100)}]

# grid search hyperparameter alpha
rnd_search_cv = RandomizedSearchCV(lasso, params, n_iter = 30, verbose = 2, cv = 3, scoring = mse, random_state = 42)

# validate search on dev data
rnd_search_cv.fit(X_dev, y_dev) # search over the dev data

# fit again on full data now with best estimator
final_model = rnd_search_cv.best_estimator_
final_model.fit(X_train_full, y_train_full); # fit lasso on full data
r2_lasso = lasso.score(X_train_full, y_train_full) # r2

# show r2 of regression model
print(round(r2_lasso, 3), 'is the r-squared for the regression model')

# generate predictions
predicted_y = lasso.predict(X_test) # prediction

# metrics
lasso_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(lasso_RMSE, 3))

# confidence interval for RMSE
lasso_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(lasso_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(final_model, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

display_scores(rmse_scores)

# append results
model_name.append('lasso')
RMSE.append(lasso_RMSE)
CI.append(lasso_ci)
cross_val_mean.append(rmse_scores.mean())

################## Linear SVM Regression ##################
# fit model
svm_reg = LinearSVR(C = 1, epsilon = 0, tol = 0.35, max_iter = 15000)
svm_reg.fit(X_train, y_train);
r2_svm = svm_reg.score(X_train, y_train) # r2

# show r2 of regression model
print(round(r2_svm, 3), 'is the r-squared for the regression model')

# setup grid search params
params = [{'C' : np.random.uniform(0.1, 200, 100), 'epsilon' : np.random.uniform(0, 5, 100)}]

# grid search hyperparameter alpha
rnd_search_cv = RandomizedSearchCV(svm_reg, params, n_iter = 30, verbose = 2, cv = 3, scoring = mse, random_state = 42)

# validate search on dev data
rnd_search_cv.fit(X_dev, y_dev) # search over the dev data

# fit again on full data now with best estimator
final_model = rnd_search_cv.best_estimator_

# fit on full data
final_model.fit(X_train_full, y_train_full);

# r2
r2_svm = final_model.score(X_train_full, y_train_full)

# show r2 of regression model
print(round(r2_svm, 3), 'is the r-squared for the regression model')

# generate predictions
predicted_y = final_model.predict(X_test) # prediction

# metrics
svm_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(svm_RMSE, 3))

# confidence interval for RMSE
svm_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(svm_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(final_model, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

# display cross-validated RMSE
display_scores(rmse_scores)

# append results
model_name.append('linear svm')
RMSE.append(svm_RMSE)
CI.append(svm_ci)
cross_val_mean.append(rmse_scores.mean())


################# XGBoost ########################
# create xgb model
xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 1,
                max_depth = 2, alpha = 2.67, n_estimators = 100, gamma = 3.80,
                booster = 'gbtree')
# set ex-post after grid search

# setup grid search params
params = {'learning_rate' : np.random.uniform(0.1, 1, 100), 'alpha': np.random.uniform(0.5, 5, 100),
 'max_depth' : [2, 3, 4, 5], 'gamma' : np.random.uniform(0.5, 5, 100), 'n_estimators': [100, 200, 300, 400, 500, 700, 1000]}

# grid search the xgboost
rnd_search_cv = RandomizedSearchCV(xgb_reg, params, n_iter = 30, verbose = 2, cv = 3, scoring = mse)

# validate search on dev data
rnd_search_cv.fit(X_dev, y_dev)

# fit again on full data now with best estimator
final_model = rnd_search_cv.best_estimator_

final_model.fit(X_train_full, y_train_full,
            eval_set=[(X_dev, y_dev)],
             early_stopping_rounds = 15)

# generate predictions
predicted_y = final_model.predict(X_test, ntree_limit = final_model.best_iteration)

# metrics
xgboost_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(xgboost_RMSE, 3))

# confidence interval for RMSE
xgb_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(xgb_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(final_model, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

# display cross-validated RMSE
display_scores(rmse_scores)

# append results
model_name.append('xgboost')
RMSE.append(xgboost_RMSE)
CI.append(xgb_ci)
cross_val_mean.append(rmse_scores.mean())


# plot feature importance
xgb.plot_importance(final_model)
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()

# get names for feature importance
x_names = df.drop(['log_price'], axis = 1)
x_names = x_names.columns.tolist()

# get dict of important variables
f_importance_dict = final_model.get_booster().get_score()

# order the dict and get top 10
top_10_f = sorted(f_importance_dict, key = f_importance_dict.get, reverse = True)[:10]

# pop 'f' off for matching variable names
top_10_f2 = []
for i in top_10_f:
    top_10_f2.append(int(i[1:]))

# find variable names for 10 top vars
top_10_f_names = []
for index, name in enumerate(x_names):
    if index in top_10_f2:
        top_10_f_names.append(name)

# ['tv', 'accommodates',  'bathrooms',  'latitude',  'longitude',
# 'miles_city_center',  'zipcode',  'bedrooms',  'room_type_Entire home/apt',
# 'room_type_Private room']

# zipcode seems redundant with lat/lon





################# XGBoost Random Forests #################
# create xgb model
xgb_reg = xgb.XGBRegressor(objective = 'reg:squarederror', learning_rate = 1,
                max_depth = 2, alpha = 2.67, n_estimators = 100, gamma = 3.80,
                booster = 'gbtree', subsample = 0.5, num_parallel_tree = 100,
                num_boost_round = 1, reg_lamba = 42.63, random_state = 42)
# set ex post from grid search

# validate on dev data
xgb_reg.fit(X_train_full, y_train_full,
            eval_set=[(X_dev, y_dev)],
             early_stopping_rounds = 15)

# generate predictions
predicted_y = xgb_reg.predict(X_test, ntree_limit = xgb_reg.best_iteration)

# metrics
xg_forest_boost_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(xg_forest_boost_RMSE, 3))

# confidence interval for RMSE
xg_forest_boost_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(xg_forest_boost_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(xgb_reg, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

# display cross-validated RMSE
display_scores(rmse_scores)

# append results
model_name.append('xg forest boost')
RMSE.append(xg_forest_boost_RMSE)
CI.append(xg_forest_boost_ci)
cross_val_mean.append(rmse_scores.mean())



#### pt summary ###
from prettytable import PrettyTable
pt = PrettyTable()
pt.field_names = ["model", "RMSE", "RMSE 95% CI", 'Cross_Val_Mean']
for i in range(0, len(model_name)):
    pt.add_row([model_name[i], np.round(RMSE[i], 3), np.round(CI[i], 3), np.round(cross_val_mean[i], 3)])
print(pt)



# setup grid search params
params = {'alpha': np.random.uniform(0.5, 5, 100), 'max_depth' : [2, 3, 4, 5],
'gamma' : np.random.uniform(0.5, 5, 100),
'n_estimators': [100, 200, 300, 400, 500, 700, 1000],
'num_parallel_tree' : [100, 200, 300],
'reg_lambda' : np.random.uniform(0.5, 50, 100)
}

# grid search the xgboost
rnd_search_cv = RandomizedSearchCV(xgb_reg, params, n_iter = 5, verbose = 2, cv = 3, scoring = mse)

# validate search on dev data
rnd_search_cv.fit(X_dev, y_dev)

# fit again on full data now with best estimator
final_model = rnd_search_cv.best_estimator_ # 397 mins of computation

final_model.fit(X_train_full, y_train_full,
            eval_set=[(X_dev, y_dev)],
             early_stopping_rounds = 15)


# generate predictions
predicted_y = final_model.predict(X_test, ntree_limit = final_model.best_iteration)

# metrics
xg_forest_boost_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(xg_forest_boost_RMSE, 3))

# confidence interval for RMSE
xg_forest_boost_ci = confidence_interval(0.95, predicted_y, y_test)
print('The 95% confidence interval for our RMSE is:', np.round(xg_forest_boost_ci, 3))

# cross-val predictive power on dev data
scores = cross_val_score(final_model, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

# display cross-validated RMSE
display_scores(rmse_scores)

# append results
model_name.append('xg forest boost')
RMSE.append(xg_forest_boost_RMSE)
CI.append(xg_forest_boost_ci)
cross_val_mean.append(rmse_scores.mean())













################# Random Forests #################

rf_reg = RandomForestRegressor(n_estimators=500, max_depth = 2, max_leaf_nodes=16, random_state=42)
rf_reg.fit(X_train, y_train)
predicted_y = rf_reg.predict(X_test)
rf_reg_error = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(rf_reg_error, 3))

bag_reg = BaggingRegressor(DecisionTreeRegressor(splitter = 'random', max_depth = 2, max_leaf_nodes=16, random_state=42), n_estimators = 300)
bag_reg.fit(X_train, y_train)
predicted_y = bag_reg.predict(X_test)
bag_reg_error = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(bag_reg_error, 3))






########################## Only top 10 Features ############################
# load csv
df = pd.read_csv('C:\\Users\\Andrew\\Desktop\\Berkeley Courses\\W207 - Applied Machine Learning\\Assignments\\Group Project\\train_cleaned.csv')
# df = df.loc[df['city'] == 'NYC']

# create dummies for some variables that should have them
df2 = pd.get_dummies(data =df, columns = ['city', 'room_type', 'property_type'])

# grab the new dummies
df2 = df2.iloc[:, 43::]

# concat into CSV DF
df = pd.concat([df, df2], axis = 1)

# remove string vars
df = df.select_dtypes(exclude=[np.object])

# drop na
df = df.dropna(axis = 0)

# prepare X, y
y = df['log_price'].values
X = df.drop(['log_price'], axis = 1)

# only top 10 features
X = X[top_10_f_names].values

# shuffle data
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

# split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state = 42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, random_state = 42)

# scale data
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)


################## Base Model: Linear Regression ##################
# fit a linear regression model
lm = LinearRegression() # instantiate lm
lm.fit(X_train_full, y_train_full); # fit lm
r2_lm = lm.score(X_train_full, y_train_full) # x, y

# show r2 of regression model
print(round(r2_lm, 3), 'is the r-squared for the regression model')

# generate predictions
predicted_y = lm.predict(X_test) # prediction

# metrics
lm_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(lm_RMSE, 3))

# confidence interval
def confidence_interval(confidence_level, predicted_y, y_test):
    confidence = confidence_level
    squared_errors = (predicted_y - y_test) ** 2
    return np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc = squared_errors.mean(),
                             scale = stats.sem(squared_errors)))

confidence_interval(0.95, predicted_y, y_test)

# cross-val
scores = cross_val_score(lm, X_dev, y_dev,
                         scoring = "neg_mean_squared_error", # expects utility fx
                         cv = 10) # 10 folds

rmse_scores = np.sqrt(-scores) # sqrt for RMSE

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", round(scores.mean(), 3))
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)


################## Linear SVM Regression ##################
# fit model
svm_reg = LinearSVR(C = 37, epsilon = 0.46, tol = 0.25, max_iter = 10000)
svm_reg.fit(X_train_full, y_train_full);
r2_svm = svm_reg.score(X_train_full, y_train_full) # r2

# show r2 of regression model
print(round(r2_svm, 3), 'is the r-squared for the regression model')

# generate results
predicted_y = svm_reg.predict(X_test) # predict values

# RMSE
svm_lm_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(svm_lm_RMSE, 3))

################## Non-Linear RBF SVM Regression ##################
# fit model
svm_rbf_reg = SVR(kernel = 'rbf', C = 36, gamma = 3.63, epsilon = 1.07)
svm_rbf_reg.fit(X_train_full, y_train_full);
r2_rbf = svm_rbf_reg.score(X_train_full, y_train_full) # r2
print(round(r2_rbf, 3), 'is the r-squared for the regression model')

# generate results
predicted_y = svm_rbf_reg.predict(X_test) # predict values

# RMSE
rbf_RMSE = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(rbf_RMSE, 3))

################# XGBoost ########################

xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1,
                max_depth = 2, alpha = 3.67, n_estimators = 500, gamma = 4.93)

xgb_reg.fit(X_train, y_train,
            eval_set=[(X_test, y_test)], early_stopping_rounds = 5)

predicted_y = xgb_reg.predict(X_test)
xgb_reg_error = mean_squared_error(y_test, predicted_y, squared = False)
print('Root Mean Squared Error:', round(xgb_reg_error, 3))
# .365

xgb.plot_importance(xgb_reg)
plt.rcParams['figure.figsize'] = [20, 10]
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


stock_data = pd.read_table('USStocks_v3.csv', sep=",")
stock_data['Date']=pd.to_datetime(stock_data['Date'], format='%Y/%m/%d')
stock_data.columns
stock_data1=stock_data[['oil_close','gold_close','bio_close','pharma_close','tech_close',
                        'energy_close', 'finance_close','health_close','airline_close','snp500_close']]
stock_data1.dtypes

##############
oil_lag_data= stock_data1.oil_close
#==================================
#  Create Lag Effects by
# adding in empty data, or zeroes
#==================================
#Lag 1 Effect
oil_lag1col = pd.Series([0])
oil_lag1col=pd.DataFrame(pd.np.append(oil_lag1col.values, oil_lag_data.values), columns=['oil_lag1col'])
oil_lag1col = oil_lag1col.ix[0:2510,]

#Lag 2 Effect
oil_lag2col = pd.Series([0,0])
oil_lag2col=pd.DataFrame(pd.np.append(oil_lag2col.values, oil_lag_data.values), columns=['oil_lag2col'])
oil_lag2col = oil_lag2col.ix[0:2510,]

##############
gold_lag_data = stock_data1.gold_close
#==================================
#  Create Lag Effects by
# adding in empty data, or zeroes
#==================================
#Lag 1 Effect
gold_lag1col = pd.Series([0])
gold_lag1col=pd.DataFrame(pd.np.append(gold_lag1col.values, gold_lag_data.values), columns=['gold_lag1col'])
gold_lag1col = gold_lag1col.ix[0:2510,]

#Lag 2 Effect
gold_lag2col = pd.Series([0,0])
gold_lag2col=pd.DataFrame(pd.np.append(gold_lag2col.values, gold_lag_data.values), columns=['gold_lag2col'])
gold_lag2col = gold_lag2col.ix[0:2510,]


#=======================================
#  Add data back into dataframe
#=======================================
goldoil_data = pd.concat([stock_data1, oil_lag1col, oil_lag2col,gold_lag1col,gold_lag2col], axis=1)
goldoil_data.head(10)
goldoil_data.tail()


#=======================
# Create Time Variable
#=======================
timelen = len(goldoil_data.index) + 1
newcols1 = pd.DataFrame({'time': list(range(1,timelen))})
goldoil_data1 = pd.concat([goldoil_data, newcols1], axis=1)
goldoil_data1.columns

#Finalized data with 2 lag effects
goldoil_data1
#=====================================
# Data splitting for time series. Do
# not randomly pull data! The data
# must be linear and incremental.
#=====================================
splitnum = np.round((len(goldoil_data1.index) * 0.5), 0).astype(int)
splitnum = np.round(((1258) * 0.3),0)
splitnum
#========================
# Split into train(50%) and test(30%) and validation (20%)
#=========================

goldoil_train_data = goldoil_data1.ix[0:1255,('oil_close','gold_close','time','oil_lag1col','oil_lag2col','gold_lag1col','gold_lag1col')]
goldoil_test_data = goldoil_data1.ix[1256:2007,('oil_close','gold_close','time','oil_lag1col','oil_lag2col','gold_lag1col','gold_lag1col')]
goldoil_vali_data = goldoil_data1.ix[2008:2510,('oil_close','gold_close','time','oil_lag1col','oil_lag2col','gold_lag1col','gold_lag1col')]

#Energy
energy_train = goldoil_data1.ix[0:1255,5]
energy_test = goldoil_data1.ix[1256:2007,5]
energy_vali = goldoil_data1.ix[2008:2510,5]

#Finance
fin_train = goldoil_data1.ix[0:1255,6]
fin_test = goldoil_data1.ix[1256:2007,6]
fin_vali = goldoil_data1.ix[2008:2510,6]

#Bio
bio_train = goldoil_data1.ix[0:1255,2]
bio_test = goldoil_data1.ix[1256:2007,2]
bio_vali = goldoil_data1.ix[2008:2510,2]

#Pharma
pharma_train = goldoil_data1.ix[0:1255,3]
pharma_test = goldoil_data1.ix[1256:2007,3]
pharma_vali = goldoil_data1.ix[2008:2510,3]

#Technology
tech_train = goldoil_data1.ix[0:1255,4]
tech_test = goldoil_data1.ix[1256:2007,4]
tech_vali = goldoil_data1.ix[2008:2510,4]

#Health
health_train = goldoil_data1.ix[0:1255,7]
health_test = goldoil_data1.ix[1256:2007,7]
health_vali = goldoil_data1.ix[2008:2510,7]

#Airline
airline_train = goldoil_data1.ix[0:1255,8]
airline_test = goldoil_data1.ix[1256:2007,8]
airline_vali = goldoil_data1.ix[2008:2510,8]

#SNP500
snp500_train = goldoil_data1.ix[0:1255,9]
snp500_test = goldoil_data1.ix[1256:2007,9]
snp500_vali = goldoil_data1.ix[2008:2510,9]



#=============================
# Standardize train and test data
#=============================

#Gold oil
scaler = preprocessing.StandardScaler()
scaler.fit(goldoil_train_data)

goldoil_train_data = scaler.transform(goldoil_train_data)
goldoil_test_data = scaler.transform(goldoil_test_data)
goldoil_vali_data = scaler.transform(goldoil_vali_data)

goldoil_train_data.describe()
##############

#Energy
scaler = preprocessing.StandardScaler()
scaler.fit(energy_train)

energy_train = scaler.transform(energy_train)
energy_test = scaler.transform(energy_test)
energy_vali = scaler.transform(energy_vali)
##############

#Finance
scaler = preprocessing.StandardScaler()
scaler.fit(fin_train)

fin_train = scaler.transform(fin_train)
fin_test = scaler.transform(fin_test)
fin_vali = scaler.transform(fin_vali)
##############

#Bio
scaler = preprocessing.StandardScaler()
scaler.fit(bio_train)

bio_train = scaler.transform(bio_train)
bio_test = scaler.transform(bio_test)
bio_vali = scaler.transform(bio_vali)
##############

#Health
scaler = preprocessing.StandardScaler()
scaler.fit(bio_train)

bio_train = scaler.transform(bio_train)
bio_test = scaler.transform(bio_test)
bio_vali = scaler.transform(bio_vali)
##############


#==============================
# Neural Network
#==============================

#Energy model1(best model)
nnts1 = MLPRegressor(activation='relu', solver='sgd', 
                      hidden_layer_sizes=(30,30))
nnts1.fit(goldoil_train_data, energy_train)

#Testing
nnpred1 = nnts1.predict(goldoil_test_data)
metrics.mean_absolute_error(energy_test, nnpred1)
metrics.mean_squared_error(energy_test, nnpred1)
metrics.r2_score(energy_test, nnpred1)


#Validation
nnpred2 = nnts1.predict(goldoil_vali_data)
metrics.mean_absolute_error(energy_vali, nnpred2)
metrics.mean_squared_error(energy_vali, nnpred2)
metrics.r2_score(energy_vali, nnpred2)
plot(nnpred2)
plot(energy_vali)

################
#Energy model2
nnts2 = MLPRegressor(activation='relu', solver='sgd')
nnts2.fit(goldoil_train_data, energy_train)

#Testing
nnpred3 = nnts2.predict(goldoil_test_data)
metrics.mean_absolute_error(energy_test, nnpred3)
metrics.mean_squared_error(energy_test, nnpred3)
metrics.r2_score(energy_test, nnpred3)

#Validation
nnpred4 = nnts2.predict(goldoil_vali_data)
metrics.mean_absolute_error(energy_vali, nnpred4)
metrics.mean_squared_error(energy_vali, nnpred4)
metrics.r2_score(energy_vali, nnpred4)


####################################
#Energy model3

nnreg1 = MLPRegressor(activation='logistic', solver='sgd', 
                      hidden_layer_sizes=(20,20), 
                      early_stopping=True)
nnreg1.fit(goldoil_train_data, energy_train)

#Testing
nnpred5 = nnreg1.predict(goldoil_test_data)
metrics.mean_absolute_error(energy_test, nnpred5)
metrics.mean_squared_error(energy_test, nnpred5)
metrics.r2_score(energy_test, nnpred5)

#Validation
nnpred6 = nnreg1.predict(goldoil_vali_data)
metrics.mean_absolute_error(energy_vali, nnpred6)
metrics.mean_squared_error(energy_vali, nnpred6)
metrics.r2_score(energy_vali, nnpred6)


#===============
#Finance models
#===============

#Finance model1
nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(goldoil_train_data, fin_train)

#Testing
nnpred1 = nnts1.predict(goldoil_test_data)
metrics.mean_absolute_error(fin_test, nnpred1)
metrics.mean_squared_error(fin_test, nnpred1)
metrics.r2_score(fin_test, nnpred1)

#Validation
nnpred2 = nnts1.predict(goldoil_vali_data)
metrics.mean_absolute_error(fin_vali, nnpred2)
metrics.mean_squared_error(fin_vali, nnpred2)
metrics.r2_score(fin_vali, nnpred2)



#Finance model(best model)
nnts2 = MLPRegressor(activation='relu', solver='sgd', 
                      hidden_layer_sizes=(3,3))
nnts2.fit(goldoil_train_data, fin_train)

#Testing
nnpred2 = nnts2.predict(goldoil_test_data)
metrics.mean_absolute_error(fin_test, nnpred2)
metrics.mean_squared_error(fin_test, nnpred2)
metrics.r2_score(fin_test, nnpred2)

#Validation
nnpred2 = nnts2.predict(goldoil_vali_data)
metrics.mean_absolute_error(fin_vali, nnpred2)
metrics.mean_squared_error(fin_vali, nnpred2)
metrics.r2_score(fin_vali, nnpred2)
plot(nnpred2)
plot(fin_vali)


###################################
#Finance model3

nnreg1 = MLPRegressor(activation='logistic', solver='sgd', 
                      hidden_layer_sizes=(20,20), 
                      early_stopping=True)
nnreg1.fit(goldoil_train_data, fin_train)

#Testing
nnpred5 = nnreg1.predict(goldoil_test_data)
metrics.mean_absolute_error(fin_test, nnpred5)
metrics.mean_squared_error(fin_test, nnpred5)
metrics.r2_score(fin_test, nnpred5)

#Validation
nnpred6 = nnreg1.predict(goldoil_vali_data)
metrics.mean_absolute_error(fin_vali, nnpred6)
metrics.mean_squared_error(fin_vali, nnpred6)
metrics.r2_score(fin_vali, nnpred6)
plot(nnpred6)
plot(fin_vali)

############################## END ############################################################
###############################################################################################


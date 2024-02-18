"""
Created on Fri Jul 21 11:45:05 2023

@author: betsa
"""

# important
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


from sklearn.preprocessing import minmax_scale

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score 

import shap

import tensorflow as tf





def create_dataset(land_use_option):
   
    years = [2006,2010, 2014, 2018, 2021]
    
    num =370* len(years)
    num_sample_per_years=[]
    X= np.zeros(shape=(num,8))
    Y= np.zeros(shape= (num,1))
    low = 0
    
    
    for year in years:
        f_name = land_use_option + str(year) +".xlsx"
        Data = pd.read_excel(f_name)
        #print(Data.info())
        #sns.pairplot(Data, y_vars='LST')
        num = len(Data)
        num_sample_per_years.append(num)
        # print("len Data:", num)
        #print("-------------------",year,"------------------")
        for i, name in enumerate(Factors):
            if name =="year":
                X[low:low+num, i] =year
            else:
                X[low:low+num, i] = Data[name]
            
        
        
        Y[low:low+num,0] = Data["LST"]
        low = low+num
    #print('---------------------------------------------------')    
    X = X[0:low,:]
    Y = Y[0:low]
    X = np.array(X)
    X = np.round(X,2)
    Y = (np.array(np.round(Y,2)))
    #X = minmax_scale(X, axis=1)
    
    return X,Y, num_sample_per_years

    
def correlation (X,Y):
    temp=np.zeros(shape = (np.shape(X)[0], np.shape(X)[1]+1))
    temp [:,0:np.shape(X)[1]] = X
    temp [:,-1] =np.reshape(Y, -1)
    corr_mat =  np.zeros(shape =(9,9))
    for i in range(9):
        for j in range(9):
            t1 =(temp[:,i])
            t2 = (temp[:,j])
            corr = np.corrcoef(t1,t2)
            corr_mat[i,j]= corr[0,1]
        
    fig1 = plt.Figure(figsize=(10,5))
    sns.heatmap(corr_mat, 
                xticklabels= ["ed","frac","lpi","lsi","pland", "x", "y", "year",'LST'],
                yticklabels=  ["ed","frac","lpi","lsi","pland", "x", "y", "year",'LST'])    
    
                   

class ML_models():
    def __init__(self, model_name, model_hparams):
        self.model_name = model_name
        self.model_hparams = model_hparams
        self.model = create_model(model_name, self.model_hparams)
 
        
def create_model(model_name, hparam):
    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators= hparam["n_estimators"],
                                      max_depth= hparam["max_depth"],
                                      criterion= hparam["criterion"])
    if model_name == 'SVM':
        model =SVR(kernel=hparam['kernel'])
        
    if  model_name == 'NN':
        model = NN_model(hparam)
        
    if model_name == 'xgboost':
        model = XGBRegressor(objective ='reg:squarederror',
                             n_estimators=hparam["n_estimators"], 
                             max_depth=hparam["max_depth"]
                             , eta=hparam["eta"], subsample=hparam["subsample"])
    if model_name == 'Dtree':
        model = DecisionTreeRegressor(max_depth= hparam["max_depth"])
        
    if model_name == 'KNN':
        model =  KNeighborsRegressor(n_neighbors = hparam["n_neighbors"])

    
    return model    


def hparam_setting(model_name):
    
    if model_name == "xgboost":
        model_hparams={"n_estimators":500,  "max_depth":4, "eta":0.1, "subsample":0.8}
                       
    if model_name == "RandomForest":
        model_hparams={"n_estimators":300,"max_depth": 8,"criterion" : "squared_error"}
        
    if model_name == 'Dtree':
        model_hparams={"max_depth": 8}
    
    if model_name =="NN":
        model_hparams ={"input": 8, 'num_layers' :5, "num_nodes":16}
    
    if model_name == 'SVM':
        model_hparams={"kernel":"rbf"}
     
    if model_name == 'KNN':
        model_hparams = {"n_neighbors":5}
        
    
    return model_hparams
        
        
def train_model(model_name, model, X, Y):
    if model_name =="NN":
        hist =model.fit(X,Y, epochs = 50, batch_size=64)
        hist= hist.history
        fig3= plt.figure(figsize=(12,6))
        plt.plot(hist['loss'])
        plt.show()
        
        
    else:
        Y= np.reshape(Y,-1)
        model = model.fit(X,Y)
    return model    

    
def test_model (model, X_Test, Y_Test):
    Y_pred = model.predict(X_Test)
    MSE = mean_squared_error(Y_Test, Y_pred)
    RMSE = math.sqrt(MSE)
    #print("RMSE", RMSE)
    return Y_pred, MSE, RMSE


def NN_model(hparam):
    num_hidden_units = hparam["num_nodes"]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_dim =hparam["input"], 
                                    units =num_hidden_units, activation = "relu"))
    
    for i in range(hparam['num_layers']):
        model.add(tf.keras.layers.Dense(units=num_hidden_units/(2**i), activation='relu'))
        #model.add(tf.keras.layers.Dropout(0.5))
        
    
    model.add(tf.keras.layers.Dense(units=1, activation = None))    
    model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss = tf.losses.MSE, 
                  metrics= ["mse", tf.metrics.RootMeanSquaredError()])     
    
    model.summary()
    
    return model
    
def visulaization(Y_True, Y_Pred):
    min_Y = np.min(Y_True)
    max_Y = np.max(Y_True)
    diagonal =[min_Y , max_Y]
    fig = plt.Figure(figsize=(12,6))
    plt.plot(diagonal, diagonal, 'r-')
    plt.plot(Y_True, Y_Pred,'bo')
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.show()
    print("hi")

def cal_AAPRE(Y,P):
    #   Yi represents the actual values.
    #   Pi represents the predicted values.
    AAPRE = np.mean(np.abs((Y-P)/Y))*100
    
    return AAPRE

def cal_Rsquare(Y,P):
    #  Yi represents the actual values.
    #  Pi represents the predicted values.
    #Avg_Y=np.mean(Y)
    #SSR= np.sum((Y-P)**2)
    #SST=np.sum((Y-Avg_Y)**2)
    #R=1-(SSR/SST)
    R=r2_score(Y, P) 
    return R
       
def cross_validation(X,Y,num_folds, model_name):
    CV = KFold(n_splits=num_folds, shuffle= True, random_state= 42)
    model_hparams = hparam_setting(model_name)
    ALL_RMSE_Train =[]
    ALL_RMSE_Test =[]
    All_AAPRE=[]
    All_R=[]
    for index_train , index_test in CV.split(X,Y):
        X_Train = X[index_train,:]
        Y_Train = Y[index_train]
        X_Test = X[index_test]
        Y_Test = Y[index_test]
       # print(X_Test[:,7])
        
        model_obj = ML_models(model_name, model_hparams)      
        model = model_obj.model
        model = train_model(model_name,model, X_Train,Y_Train)  
        
        Y_pred_Train, MSE, RMSE_Train = test_model(model, X_Train,Y_Train)
        ALL_RMSE_Train.append(RMSE_Train)
        
        Y_pred_Test, MSE, RMSE_Test = test_model(model, X_Test,Y_Test)
        ALL_RMSE_Test.append(RMSE_Test)
        
        All_AAPRE.append(cal_AAPRE(Y_Test,Y_pred_Test))
        All_R.append(cal_Rsquare(Y_Test,Y_pred_Test))
        
        visulaization(Y_Test, Y_pred_Test)
        
    
    for i in range(num_folds):
        print("fold ", i, ":")
        print("RMSE Test=", ALL_RMSE_Test[i])
        print("RMSE Train=", ALL_RMSE_Train[i])
        print("AAPRE=",All_AAPRE[i])
        print("R=",All_R[i])
        
    print('--------------------------------------------------------------')   
    print("AVG Train RMSE =", np.mean(ALL_RMSE_Train), "+/- =", np.std(ALL_RMSE_Train))
    print("AVG Test RMSE =", np.mean(ALL_RMSE_Test), "+/- =", np.std(ALL_RMSE_Test) ) 
    print("AVG Test AAPRE=", np.mean(All_AAPRE)) 
    print("AVG Test R=", np.mean(All_R)) 
    
 
def train_test_2021(X,Y,num_sample_per_years, model_name):
   
    num_samples= len(Y)
    num_test_samples = num_sample_per_years[-1]
    print(num_test_samples)
    num_train_samples = num_samples - num_test_samples
    X_Train = X[0:num_train_samples,:]
    Y_Train = Y[0:num_train_samples]
    X_Test = X[num_train_samples+1:num_samples,:]
    Y_Test = Y[num_train_samples+1:num_samples]
    print(np.unique(X_Test[:,-1]))
    print(np.unique(X_Train[:,-1]))
    
    model_hparams = hparam_setting(model_name)    
    model_obj = ML_models(model_name, model_hparams)      
    model = model_obj.model
    model = train_model(model_name,model, X_Train,Y_Train)  
        
    Y_pred_Train, MSE, RMSE_Train = test_model(model, X_Train,Y_Train)
      
    Y_pred_Test, MSE, RMSE_Test = test_model(model, X_Test,Y_Test)
        
        
    All_AAPRE =cal_AAPRE(Y_Test,Y_pred_Test)
    All_R=cal_Rsquare(Y_Test,Y_pred_Test)
        
    visulaization(Y_Test, Y_pred_Test)
    print("--------------------------------------------------------------")
    print("----------------Result test 2021--------------------------------------")
    print("RMSE  Train=", RMSE_Train)
    print("RMSE  Test=", RMSE_Test)    
    print("AAPRE Test=", All_AAPRE)    
    print("R Test=", All_R)    
    
    
    
 
def feature_importance(X,Y, model_name):
    print("hi")
    model_hparams = hparam_setting(model_name)
    model_obj = ML_models(model_name, model_hparams)      
    model = model_obj.model
    
    #split train and test
    X_Train, X_Test, Y_Train, Y_Test =  train_test_split(X,Y, test_size=0.2)
    
    
    if model_name  == "RandomForest":
        explainer = shap.TreeExplainer(model)
        model = model.fit(X_Train, Y_Train)
        
        shap_values = explainer.shap_values(X_Train)
        fig1 = plt.Figure(figsize= (12,6))
        shap.summary_plot(shap_values, X_Train, plot_type= "bar", feature_names = Factors)
        print("hi first shap")

        fig2 = plt.Figure(figsize=(12,6))
        shap.summary_plot(shap_values, X_Train, feature_names= Factors)
        
        print("hi end")
    if model_name == 'NN':
        explainer = shap.KernelExplainer(model.predict,X_Train)
        hist = model.fit(X_Train, Y_Train, batch_size= 64, epochs =50)
        
        shap_values = explainer.shap_values(X_Train, nsamples =5)
        fig1 = plt.Figure(figsize= (12,6))
        shap.summary_plot(shap_values, X_Train, plot_type= "bar", feature_names = Factors)
        print("hi first shap")

        fig2 = plt.Figure(figsize=(12,6))
        shap.summary_plot(shap_values, X_Train, feature_names= Factors)
        print("hi end")
    
    

    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#def main():
Factors= ["ed","frac","lpi","lsi","pland", "x", "y", "year"]   
# 1- create dataset 
##buildup =allb
configuration_mode =['allb','allv','alls'] 
X,Y, num_sample_per_years = create_dataset(configuration_mode[2])

# 2- Cross-validation
model_names= ["RandomForest", "NN","SVM","xgboost","Dtree", "KNN"]
cross_validation(X, Y, num_folds =5 , model_name =model_names[0])

# 3- Test 2021
train_test_2021(X,Y,num_sample_per_years, model_names[0])

#feature_importance(X,Y, model_name =model_names[0])
#correlation(X, Y)    

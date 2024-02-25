'''@author: betsa
"""

"""
Created on Fri Jul 21 11:45:05 2023

@author: betsa
"""
'''

# important
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from skgarden import RandomForestQuantileRegressor

import pandas as pd
import seaborn as sns
import math


from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost
from sklearn.metrics import r2_score 
from sklearn.svm import SVR
import xgboost as xg
from xgboost import XGBRegressor
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import AdaBoostRegressor

import tensorflow as tf
from keras.regularizers import l2

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
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
    #print(len(Y))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.round(X,2)
    #X = minmax_scale(X, axis=1)
    #X, Y = oulier_removal(X,Y)
    
    return X,Y, num_sample_per_years

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def oulier_removal(X,Y):
    list_remove_index =[]
    for i in range(len(Factors)):
        Q1 = np.percentile(X[:,i], 25, interpolation = 'midpoint') 
        Q2 = np.percentile(X[:,i], 50, interpolation = 'midpoint') 
        Q3 = np.percentile(X[:,i], 75, interpolation = 'midpoint') 
         
        print('-------------------------Feature=', Factors[i],'---------------------------------')
        print('Q1 25 percentile of the given data is, ', Q1)
        print('Q1 50 percentile of the given data is, ', Q2)
        print('Q1 75 percentile of the given data is, ', Q3)
     
        IQR = Q3 - Q1 
        k = 1.5
        print('Interquartile range is', IQR)
        low_lim = Q1 - k * IQR
        up_lim = Q3 + k* IQR
        
        #fig3= plt.figure(figsize =(10, 4))
        #plt.boxplot(X[:,i],vert= False)
        #plt.axvline(low_lim, color ='r', label="low" )
        #plt.axvline(up_lim, color ='r', label="up") 
        #plt.title("box plot " +str(Factors[i]))
        #plt.show()
        
        outlier =[]
        temp = X[:,i]
        index_list=[]
        for x,j in zip(temp,range(1357)):
           if ((x> up_lim) or (x<low_lim)):
              outlier.append(x)
              index_list.append(j)
        print(' outlier in the dataset is', outlier)
        print("number of outlier :", len(outlier))
        print("outlier index:", index_list)
        list_remove_index = np.union1d(list_remove_index, index_list)
    
    
    
    NewX = []
    NewY = []
    for i in range(len(Y)):
        if i in list_remove_index:
             continue
        else:
             NewX.append( X[i,:])
             NewY.append( Y[i])
             
    NewX = np.array(NewX)
    NewY = np.array(NewY) 

    return NewX, NewY         
             
             
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------                  

class ML_models():
    def __init__(self, model_name, model_hparams):
        self.model_name = model_name
        self.model_hparams = model_hparams
        self.model = create_model(model_name, self.model_hparams)
 
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
def train_model(model_name, model, X, Y):
    if model_name =="NN":
        hist =model.fit(X,Y, epochs = 1500, batch_size=256)
        hist= hist.history
        #print(np.shape(hist['loss']))
        myfig= plt.figure(figsize=(12,6))
        plt.plot(hist['loss'][100:])
        plt.show()
        
    else:
        print("Train")
        Y= np.reshape(Y,-1)
        model = model.fit(X,Y)
    return model   
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
   
def test_model (model, X_Test, Y_Test):
    Y_pred = model.predict(X_Test)
    MSE = mean_squared_error(Y_Test, Y_pred)
    RMSE = math.sqrt(MSE)
    #print("RMSE", RMSE)
    return Y_pred, MSE, RMSE

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def NN_model(hparam):
    num_hidden_units = hparam["num_nodes"]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(input_dim =hparam["input"], 
                                    units =num_hidden_units, activation = "relu"))
    
    
    #for i in range(hparam['num_layers']):
     #   model.add(tf.keras.layers.Dense(units=num_hidden_units*(2**(i+1)), activation='relu', activity_regularizer=l2(0.001)))
        #model.add(tf.keras.layers.Dropout(0.5))'''
        
    
    #temp = num_hidden_units*(2**i)
    for i in range(hparam['num_layers']):
        model.add(tf.keras.layers.Dense(units=num_hidden_units/(2**(i+1)), activation='relu', name = 'layer' + str(i)
                                     ,  activity_regularizer=(l2(0.005))))
        #model.add(tf.keras.layers.Dropout(0.6))
        #model.add(tf.keras.layers.BatchNormalization())
        
    
    model.add(tf.keras.layers.Dense(units=1, activation = None))    
    model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss = tf.losses.MSE,
                  metrics= ["mse", tf.metrics.RootMeanSquaredError()])     
    
    model.summary()
    #pdb.set_trace()
    
    return model
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------        
def create_model(model_name, hparam):
  
    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators= hparam["n_estimators"],
                                      max_depth= hparam["max_depth"],
                                      criterion= hparam["criterion"])
    if  model_name == 'NN':
        model = NN_model(hparam)
    
        
    if model_name == 'DTree':
         model = tree.DecisionTreeRegressor(criterion= hparam['criterion'], 
                                            max_depth=hparam['max_depth'],
                                            min_samples_split= hparam['min_samples_split'])
         
        
    if model_name== 'Adaboost':
        base = tree.DecisionTreeRegressor(criterion=hparam['criterion'],
                                          max_depth=hparam['max_depth'],
                                          min_samples_split= hparam['min_samples_split'])
        
        model = AdaBoostRegressor(n_estimators=hparam['n_estimators'],
                                  base_estimator =base)
    
    if model_name  == 'xgboost':
        print("xgboost")
        model = xg.XGBRegressor(objective =hparam['criterion'],
                                max_depth=hparam['max_depth'],
                                n_estimators = hparam['n_estimators'],
                                eta = hparam["eta"],
                                subsample =hparam["subsample"],
                                seed = 123)
    if model_name == 'KNN':
        model =  neighbors.KNeighborsRegressor(n_neighbors =hparam['n_neighbors'])
        
    if model_name  == 'SVR':
        model =SVR ()
    
    return model    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def hparam_setting(model_name):
    if model_name == "RandomForest":
        model_hparams={"n_estimators":100,"max_depth":10,
                       "criterion" : "squared_error"}
    
    if model_name == "xgboost":
        print("parm")
        model_hparams={"n_estimators":1000,"max_depth":4,
                       "criterion" : 'reg:squarederror', 
                       "eta" :0.05,
                       "subsample":0.8}
    
    if model_name == 'DTree':
        model_hparams={"max_depth":8,
                       "criterion" : "squared_error", 
                       "min_samples_split":20}
        
    if model_name == 'Adaboost':
        model_hparams={"n_estimators":100, "max_depth":8,
                       "criterion" : "squared_error",
                       "min_samples_split":10}
    
    if model_name =="NN":
        model_hparams ={"input": input_dim, 'num_layers' :3, "num_nodes":128}
        
    if model_name == 'KNN':
        model_hparams ={'n_neighbors':15}
    
    if model_name == 'SVR':
       model_hparams ={}
       
    
    return model_hparams
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
def Quantile_Regression_Forests(X,Y):
    kf = KFold(n_splits=5, random_state=0)
    rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=10, n_estimators=100)
    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
def mse_boxplot(X,Y):
    #split train and test
    #X_Train, X_Test, Y_Train, Y_Test =  train_test_split(X,Y, test_size=0.2, shuffle=True)
    
    num_samples= len(Y)
    print("total number of samples:", num_samples)
    
    num_test_samples = num_sample_per_years[-1]
    print("number of test=",num_test_samples)
    
    num_train_samples = num_samples - num_test_samples
    X_Train = X[0:num_train_samples,:]
    Y_Train = Y[0:num_train_samples]
    X_Test = X[num_train_samples:num_samples,:]
    Y_Test = Y[num_train_samples:num_samples]
    
    
    
    #--------------------------------------
    model_names= ["RandomForest", "NN","SVR","xgboost","DTree", "KNN","Adaboost"]
    mse_values= []
 
    MSE_all_points =np.zeros(shape=(len(Y_Test),len(model_names)))
    i=0
    
    for algorithm in model_names:
        model_hparams = hparam_setting(algorithm)
        model_obj = ML_models(algorithm, model_hparams)      
        model = model_obj.model
        model = train_model(algorithm,model, X_Train,Y_Train)  
        Y_pred_Test, MSE, RMSE_Test = test_model(model, X_Test,Y_Test)
        # List of MSEs for each model
        mse_values.append(MSE)
        Y_pred_Test =np.reshape(Y_pred_Test, np.shape(Y_Test))
        all_mse= (Y_pred_Test-Y_Test)**2
        all_mse = np.reshape(all_mse, (len(Y_Test)))
        MSE_all_points[:,i]=all_mse
        i = i+1
       
        

    # Boxplot
    #bp = ax.boxplot(MSE_all_points)
    plt.boxplot(MSE_all_points)
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.title('MSEs from Predictions of Different Models')
    plt.xticks(ticks=range(1, len(mse_values) + 1),labels =model_names)
    plt.show()
    print("MSE =", mse_values)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Factors= ["ed","frac","lpi","lsi","pland", "x", "y", "year"]  
input_dim = len(Factors) 
# 1- create dataset 
configuration_mode =['allb','allv','alls'] 
X,Y, num_sample_per_years = create_dataset(configuration_mode[1])
    
#2- MSE box plot
mse_boxplot(X,Y)

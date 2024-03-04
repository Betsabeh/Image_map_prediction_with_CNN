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
import pdb

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler 


import xgboost as xg
from xgboost import XGBRegressor
import lightgbm as lgb
from ngboost import ngboost # NGBoost
from ngboost import learners # default_tree_learner
from ngboost import distns # Normal
from ngboost import scores # MLE

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 

import shap
import properscoring as ps


from scipy import stats
from scipy.stats.mstats import mquantiles

from plotnine import *

import tensorflow as tf
from keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow.keras import layers # Input, Dense, Concatenate
from tensorflow.keras import models # Model
from tensorflow.keras import optimizers # Adam

tfd = tfp.distributions
tfb = tfp.bijectors
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
        print("-------------------",year,"------------------")
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
    X, Y = oulier_removal(X,Y)
    
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
        model_hparams ={"input": input_dim, 'num_layers' :1, "num_nodes":128}
        
    if model_name == 'KNN':
        model_hparams ={'n_neighbors':15}
    
    if model_name == 'SVR':
       model_hparams ={}
       
    
    return model_hparams
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
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
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def cal_AAPRE(Y,P):
    #   Yi represents the actual values.
    #   Pi represents the predicted values.
    AAPRE = np.mean(np.abs((Y-P)/Y))*100
    
    return AAPRE

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def cal_Rsquare(Y,P):
    #  Yi represents the actual values.
    #  Pi represents the predicted values.
    #Avg_Y=np.mean(Y)
    #SSR= np.sum((Y-P)**2)
    #SST=np.sum((Y-Avg_Y)**2)
    #R=1-(SSR/SST)
    R=r2_score(Y, P) 
    return R

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def cal_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair is not 0:
        return summ/pair
    else:
        return 0    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def cross_validation(X,Y,num_folds, model_name):
    CV = KFold(n_splits=num_folds, shuffle= True, random_state= 42)
    model_hparams = hparam_setting(model_name)
    ALL_RMSE_Train =[]
    ALL_RMSE_Test =[]
    All_AAPRE=[]
    All_R=[]
    CI =[]
    for index_train , index_test in CV.split(X,Y):
        X_Train = X[index_train,:]
        Y_Train = Y[index_train]
        X_Test = X[index_test]
        Y_Test = Y[index_test]
       # print(X_Test[:,7])
        
        model_obj = ML_models(model_name, model_hparams)      
        model = model_obj.model
        model = train_model(model_name,model, X_Train,Y_Train)  
        print("len Train", len(Y_Train))
        
        Y_pred_Train, MSE, RMSE_Train = test_model(model, X_Train,Y_Train)
        ALL_RMSE_Train.append(RMSE_Train)
        
        Y_pred_Test, MSE, RMSE_Test = test_model(model, X_Test,Y_Test)
        ALL_RMSE_Test.append(RMSE_Test)
        
        All_AAPRE.append(cal_AAPRE(Y_Test,Y_pred_Test))
        All_R.append(cal_Rsquare(Y_Test,Y_pred_Test))
        CI.append(cal_cindex(Y_Test, Y_pred_Test))
        
        visulaization(Y_Test, Y_pred_Test)
        
    
    for i in range(num_folds):
        print("fold ", i, ":")
        print("RMSE Test=", ALL_RMSE_Test[i])
        print("RMSE Train=", ALL_RMSE_Train[i])
        print("AAPRE=",All_AAPRE[i])
        print("R=",All_R[i])
        print("Test CI =", CI[i])
        
    print('--------------------------------------------------------------')   
    print("AVG Train RMSE =", np.mean(ALL_RMSE_Train), "+/- =", np.std(ALL_RMSE_Train))
    print("AVG Test RMSE =", np.mean(ALL_RMSE_Test), "+/- =", np.std(ALL_RMSE_Test) ) 
    print("AVG Test AAPRE=", np.mean(All_AAPRE)) 
    print("AVG Test R=", np.mean(All_R)) 
    print("AVG CI:", np.mean(CI))
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def nll_loss(y, distr):
    return -distr.log_prob(y) 

def model_distribution(params): 
    return tfd.Normal(loc=params[:,0:1], scale=tf.math.softplus(params[:,1:2]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_predicted_distribution(index, figsize = (25, 7)):
    row = comparison_df.iloc[index]
    
    true_value = row["true"]
    loc1 = row["mean_dl"]
    loc2 = row["mean_ngboost"]
    
    scale1 = row["sd_dl"]
    scale2 = row["sd_ngboost"]
    
    
    nll_dl = -stats.norm.logpdf(true_value, loc = loc1, scale = scale1)
    nll_ngboost = -stats.norm.logpdf(true_value, loc = loc2, scale = scale2)
    ############################################################################
    x = np.linspace(min(y_test), max(y_test), 100)

    colors = ["#348ABD", "#A60628"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)
    fig.suptitle('Comparison between DL and NGBoost')

    pdf1 = stats.norm.pdf(x, loc = loc1, scale=scale1)
    ax1.plot(x, pdf1, lw=3, color=colors[0], label=f"$\mu$={np.round(loc1,4)}, $\sigma$={np.round(scale1, 4)}, \ntrue obs = {true_value}")
    ax1.fill_between(x, stats.norm.pdf(x, loc = loc1, scale = scale1), color=colors[0], alpha=.33)
    ax1.legend()
    ax1.set_ylabel("PDF")
    #ax1.set_xlabel("$x$")
    ax1.set_title(f"DL: NLL = {np.round(nll_dl, 2)}")
    ax1.set_xlim(true_value/2, true_value * 2)
    ax1.vlines(true_value, 0, max(pdf1), 'g', linestyle = "--", label='true label')

    pdf2 = stats.norm.pdf(x, loc = loc2, scale=scale2)
    ax2.plot(x, pdf2, lw=3, color=colors[1], label=f"$\mu$={np.round(loc2, 4)}, $\sigma$={np.round(scale2, 4)}, \ntrue obs = {true_value}")
    ax2.fill_between(x, stats.norm.pdf(x, loc = loc2, scale = scale2), color=colors[1], alpha=.33)
    ax2.legend()
    ax2.set_ylabel("PDF")
    #ax2.set_xlabel("$x$")
    ax2.set_title(f"NGBoost: NLL = {np.round(nll_ngboost,2)}")
    ax2.set_xlim(true_value/2, true_value * 2)
    ax2.vlines(true_value, 0, max(pdf2), 'g', linestyle = "--", label='true label')
    
    
    
    ax1.title.set_fontsize(30)
    ax1.xaxis.label.set_fontsize(20)
    ax1.yaxis.label.set_fontsize(20)
    
    ax2.title.set_fontsize(30)
    ax2.xaxis.label.set_fontsize(20)
    ax2.yaxis.label.set_fontsize(20)
    
    plt.show()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def uncertainity(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    
    #* 1- Quantile Regression
    quantiles = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    quantile_predictions = {}
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    
    params = {'objective': 'regression'}
    lgb_model = lgb.train(params=params, train_set=train_data, num_boost_round=100)
    lgb_prediction = lgb_model.predict(X_test)

    #train models on quantiles
    for quantile in quantiles:
      print(f"modeling quantile {quantile}")
      params = {'objective': 'quantile', 'alpha': quantile}
      lgb_model = lgb.train(params=params, train_set=train_data, num_boost_round=100)
      pred = lgb_model.predict(X_test)
    
    quantile_predictions[quantile] = pred
    #lets check LightGBM RMSE
    mse_LGB = mean_squared_error(y_true = y_test, y_pred = lgb_prediction, squared=False)
    print("LGBOOST RMSE=", mse_LGB)
    empirical_quantiles = []
    for quantile in quantiles:
       empirical = (quantile_predictions[quantile] >= y_test).mean()
       empirical_quantiles.append(empirical)
   
    pd.DataFrame({'quantile': quantiles, 'empirical_quantiles': empirical_quantiles})
    plt.figure(figsize=(16, 10))
    sns.set_context("notebook", font_scale=2)
    sns.lineplot(x = quantiles, y = quantiles, color = "magenta", linestyle='--', linewidth=3, label = "ideal")
    sns.lineplot(x = quantiles, y = empirical_quantiles, color = "black", linestyle = "dashdot", linewidth=3, label = "observed")
    sns.scatterplot(x = quantiles, y = empirical_quantiles, marker="o", s = 150)
    plt.legend()
    plt.xlabel("True Quantile")
    plt.ylabel("Empirical Quantile")
    _ = plt.title("Reliability diagram: assessment of quantile predictions")
    
    
    #*  2- NGboost
    ngb = ngboost.NGBoost(Base=learners.default_tree_learner, Dist=distns.Normal, Score=scores.LogScore, natural_gradient=True, verbose=True)
    ngb.fit(X_train, y_train)

    #predicted mean
    ngb_mean_pred = ngb.predict(X_test)

    #predicted distribution
    ngb_dist_pred = ngb.pred_dist(X_test)
    
    #let's check NGBoost RMSE
    mse_NGBOOST =mean_squared_error(y_true = y_test, y_pred = ngb_mean_pred, squared=False)
    print("NGBOOST RMSE=", mse_NGBOOST )
    nll_ngboost = -ngb_dist_pred.logpdf(y_test)
    print(f"NLL NGBoost: {nll_ngboost.mean()}")
    
    #Compare NLL
    print(np.log(stats.norm.pdf(y_test.values[0], loc = 0.477102760902954, scale = 0.0022844303231773434)))
    
    
    #* 3-Deep Learning Probabilistic Prediction
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    

    inputs = layers.Input(shape=((len(X_test.columns),)))

    hidden1 = layers.Dense(100, activation = "relu", name = " dense_mean_1")(inputs)
    hidden2 = layers.Dense(50, activation = "relu", name = "dense_mean_2")(hidden1)
    output_mean = layers.Dense(1, name = "mean_output")(hidden2) #expected mean


    hidden1 = layers.Dense(100,activation="relu", name = "dense_sd_1")(inputs)
    hidden1 = layers.Dropout(0.1)(hidden1)
    hidden2 = layers.Dense(50,activation="relu", name = "dense_sd_2")(hidden1)
    hidden2 = layers.Dropout(0.1)(hidden2)
    hidden3 = layers.Dense(20,activation="relu", name = "dense_sd_3")(hidden2)
    output_sd = layers.Dense(1, name = "sd_output")(hidden3)

    mean_sd_layer = layers.Concatenate(name = "mean_sd_concat")([output_mean, output_sd]) 
    dist = tfp.layers.DistributionLambda(model_distribution)(mean_sd_layer) 

    dist_mean = tfp.layers.DistributionLambda( make_distribution_fn=model_distribution, convert_to_tensor_fn=tfp.distributions.Distribution.mean)(mean_sd_layer)
    dist_std = tfp.layers.DistributionLambda( make_distribution_fn=model_distribution, convert_to_tensor_fn=tfp.distributions.Distribution.stddev)(mean_sd_layer)

    model_distr = models.Model(inputs=inputs, outputs=dist)
    model_distr.compile(optimizers.Adagrad(learning_rate=0.001), loss=nll_loss)
    model_mean = models.Model(inputs=inputs, outputs=dist_mean)
    model_sd = models.Model(inputs=inputs, outputs=dist_std)
    model_distr.summary()
    history = model_distr.fit(X_train_scaled, y_train, epochs=150, verbose=1, batch_size = 2**7, validation_data=(X_test_scaled,y_test))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylabel('NLL')
    plt.xlabel('Epochs')
    plt.show()
    
    print(model_distr.evaluate(X_train_scaled,y_train, verbose=0))
    print(model_distr.evaluate(X_test_scaled, y_test, verbose=0))
    dl_mean_prediction  = model_mean.predict(X_test_scaled).reshape(-1)
    dl_sd_prediction = model_sd.predict(X_test_scaled).reshape(-1)
    mean_squared_error(y_true = y_test, y_pred=model_mean.predict(X_test_scaled), squared = False)
    nll_dl = []
    for (true_mean, mean_temp, sd_temp) in zip(y_test, dl_mean_prediction, dl_sd_prediction):
      nll_temp = -stats.norm.logpdf(true_mean, loc = mean_temp, scale = sd_temp)
      nll_dl.append(nll_temp)
    
    comparison_df = pd.DataFrame({'nll_ngboost': nll_ngboost, 'nll_dl': nll_dl, 
                               'true': y_test, 'mean_dl': dl_mean_prediction, 'mean_ngboost': ngb_mean_pred, 
                               'sd_dl': dl_sd_prediction, 'sd_ngboost': ngb_dist_pred.params["scale"]
                              }).reset_index(drop=True)
    
    
    plot_predicted_distribution(0)
    #check if the true value is within the predictive interval

    validity_pd = pd.DataFrame()
    for interval in [0.6, 0.7, 0.8, 0.9]:
      #calculate density concentration 
      interval = np.round(interval, 1)
      validity_ngboost = stats.norm.interval(interval, loc = ngb_mean_pred, scale = ngb_dist_pred.params["scale"])
      validity = np.mean((validity_ngboost[0] > y_test) | (y_test > validity_ngboost[1]))
      validity_pd = pd.concat([validity_pd, pd.DataFrame({'estimator': ['NGBoost'], 'interval': [interval], 'validity': [validity]})])  

    for interval in [0.6, 0.7, 0.8, 0.9]:
      validity_dl = stats.norm.interval(interval, loc = dl_mean_prediction, scale = dl_sd_prediction)
      validity = np.mean((validity_dl[0] > y_test) | (y_test > validity_dl[1]))
      validity_pd = pd.concat([validity_pd, pd.DataFrame({'estimator': ['DL'], 'interval': [interval], 'validity': [validity]})])  
    
    for quantile in [(0.2, 0.8), (0.15, 0.85), (0.1, 0.9), (0.05, 0.95)]:
       interval = np.round(quantile[1] - quantile[0], 1)
       validity = np.mean((quantile_predictions[quantile[0]] > y_test) | (y_test > quantile_predictions[quantile[1]]))
       validity_pd = pd.concat([validity_pd, pd.DataFrame({'estimator': ['Lightgbm Quantile Regression'], 'interval': [interval], 'validity': [validity]})]) 
    
    validity_pd = validity_pd.reset_index(drop=True).sort_values(by="interval").set_index(["estimator", "interval"])
    print(validity_pd)
    
    ggplot(aes("estimator", "validity", fill = "estimator"), validity_pd.reset_index()) + geom_bar(stat = "identity") + facet_grid(".~interval") + \
    xlab("") + ylab("") + ggtitle("Bias at different intervals") + theme_seaborn(font_scale = 2) +  \
    theme(figure_size=(16, 6), axis_text_x=element_text(rotation=45, hjust=1))

   

   # sharpness
    sharpness_pd = pd.DataFrame()
    for interval in [0.6, 0.7, 0.8, 0.9]:
       #calculate density concentration 
       interval = np.round(interval, 1)
       sharpness_ngboost = stats.norm.interval(interval, loc = ngb_mean_pred, scale = ngb_dist_pred.params["scale"])
       value = np.mean(sharpness_ngboost[1] - sharpness_ngboost[0])
       sharpness_pd = pd.concat([sharpness_pd, pd.DataFrame({'estimator': ['NGBoost'], 'interval': [interval], 'value': [value]})])  

    for interval in [0.6, 0.7, 0.8, 0.9]:
       sharpness_dl = stats.norm.interval(interval, loc = dl_mean_prediction, scale = dl_sd_prediction)
       value = np.mean(sharpness_dl[1] - sharpness_dl[0])
       sharpness_pd = pd.concat([sharpness_pd, pd.DataFrame({'estimator': ['DL'], 'interval': [interval], 'value': [value]})])  
    
    for quantile in [(0.2, 0.8), (0.15, 0.85), (0.1, 0.9), (0.05, 0.95)]:
      interval = np.round(quantile[1] - quantile[0], 1)
      value = (quantile_predictions[quantile[1]] - quantile_predictions[quantile[0]]).mean()
      sharpness_pd = pd.concat([sharpness_pd, pd.DataFrame({'estimator': ['Lightgbm Quantile Regression'], 'interval': [interval], 'value': [value]})]) 

    sharpness_pd = sharpness_pd.reset_index(drop=True).sort_values(by="interval").set_index(["estimator", "interval"])
    print(sharpness_pd)
    
    gplot(aes("estimator", "value", fill = "estimator"), sharpness_pd.reset_index()) + geom_bar(stat = "identity") + facet_grid(".~interval", scales = "free") + \
    xlab("") + ylab("") + ggtitle("Sharpness at different intervals") + theme_seaborn(font_scale = 2) +  \
    theme(figure_size=(16, 6), axis_text_x=element_text(rotation=45, hjust=1))
    
    comparison_df["crps_ngboost"] = comparison_df.apply(lambda x: ps.crps_gaussian(x["true"], mu = x["mean_ngboost"], sig = x["sd_ngboost"]), axis = 1)
    comparison_df["crps_dl"] = comparison_df.apply(lambda x: ps.crps_gaussian(x["true"], mu = x["mean_dl"], sig = x["sd_dl"]), axis = 1)
    
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------     
 
    
    

    
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#def main():
Factors= ["ed","frac","lpi","lsi","pland", "x", "y", "year"]  
input_dim = len(Factors) 
# 1- create dataset 
##buildup =allb
configuration_mode =['allb','allv','alls'] 
X,Y, num_sample_per_years = create_dataset(configuration_mode[0])

uncertainity(X,Y)

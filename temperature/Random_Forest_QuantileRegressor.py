
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

from sklearn.svm import SVR
import xgboost as xg
from xgboost import XGBRegressor
from sklearn import tree
from sklearn import neighbors
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score 

import shap

import tensorflow as tf
from keras.regularizers import l2
import quantile_forest
from quantile_forest import RandomForestQuantileRegressor
import altair as alt

#-----------------------------------------------------------------------------


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
        model_hparams ={"input": input_dim, 'num_layers' :3, "num_nodes":128}
        
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
def Random_Forest_QuantileRegressor(X,Y,num_folds):
    
    qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=0)
    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(X)
    
    # Using k-fold cross-validation, get predictions for all samples.
    data = {"y_true": [], "y_pred": [], "y_pred_low": [], "y_pred_upp": []}
    for train_index, test_index in kf.split(X):
       X_train, y_train = X[train_index,:], Y[train_index]
       X_test, y_test = X[test_index,:], Y[test_index]
       
       qrf.set_params(max_features=X_train.shape[1] // 3)
       qrf.fit(X_train, y_train)

       # Get predictions at 95% prediction intervals and median.
       y_pred_i = qrf.predict(X_test, quantiles=[0.025, 0.5, 0.975])

       data["y_true"].extend(y_test)
       data["y_pred"].extend(y_pred_i[:, 1])
       data["y_pred_low"].extend(y_pred_i[:, 0])
       data["y_pred_upp"].extend(y_pred_i[:, 2])
    
    df = pd.DataFrame(data).pipe(lambda x: x )
    
    return df  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def plot_calibration_and_intervals(df):
    def plot_calibration(df):
        domain = [
            int(np.min(np.minimum(df["y_true"], df["y_pred"]))),  # min of both axes
            int(np.max(np.maximum(df["y_true"], df["y_pred"]))),  # max of both axes
        ]

        tooltip = [
            alt.Tooltip("y_true:Q", format="$,d", title="Actual Price"),
            alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price"),
            alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
            alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
        ]

        base = alt.Chart(df)

        circle = base.mark_circle(size=30).encode(
            x=alt.X(
                "y_pred:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="Fitted Values (conditional median)",
            ),
            y=alt.Y(
                "y_true:Q",
                axis=alt.Axis(format="$,d"),
                scale=alt.Scale(domain=domain, nice=False),
                title="Observed Values",
            ),
            color=alt.value("#f2a619"),
            tooltip=tooltip,
        )

        bar = base.mark_bar(opacity=0.8, width=2).encode(
            x=alt.X("y_pred:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y=alt.Y("y_pred_low:Q", scale=alt.Scale(domain=domain, padding=0), title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=alt.value("#e0f2ff"),
            tooltip=tooltip,
        )

        tick = base.mark_tick(opacity=0.4, orient="horizontal", thickness=1, width=5).encode(
            x=alt.X("y_pred:Q", title=""), color=alt.value("#006aff")
        )
        tick_low = tick.encode(y=alt.Y("y_pred_low:Q", title=""))
        tick_upp = tick.encode(y=alt.Y("y_pred_upp:Q", title=""))

        diagonal = (
            alt.Chart(
                pd.DataFrame({"var1": [domain[0], domain[1]], "var2": [domain[0], domain[1]]})
            )
            .mark_line(color="black", opacity=0.4, strokeDash=[2, 2])
            .encode(
                x=alt.X("var1:Q"),
                y=alt.Y("var2:Q"),
            )
        )

        chart = bar + tick_low + tick_upp + circle + diagonal
        return chart

    def plot_intervals(df):
        df = df.copy()

        # Order samples by interval width.
        y_pred_interval = df["y_pred_upp"] - df["y_pred_low"]
        sort_idx = np.argsort(y_pred_interval)
        df = df.iloc[sort_idx]
        df["idx"] = np.arange(len(df))

        # Center data, with the mean of the prediction interval at 0.
        mean = (df["y_pred_low"] + df["y_pred_upp"]) / 2
        df["y_true"] -= mean
        df["y_pred"] -= mean
        df["y_pred_low"] -= mean
        df["y_pred_upp"] -= mean

        x_domain = [0, len(df)]
        y_domain = [
            int(np.min(np.minimum(df["y_true"], df["y_pred"]))),  # min of both axes
            int(np.max(np.maximum(df["y_true"], df["y_pred"]))),  # max of both axes
        ]

        tooltip = [
            alt.Tooltip("idx:Q", format=",d", title="Sample Index"),
            alt.Tooltip("y_true:Q", format="$,d", title="Actual Price (Centered)"),
            alt.Tooltip("y_pred:Q", format="$,d", title="Predicted Price (Centered)"),
            alt.Tooltip("y_pred_low:Q", format="$,d", title="Predicted Lower Price"),
            alt.Tooltip("y_pred_upp:Q", format="$,d", title="Predicted Upper Price"),
            alt.Tooltip("y_pred_width:Q", format="$,d", title="Prediction Interval Width"),
        ]

        base = alt.Chart(df).transform_calculate(
            y_pred_width=alt.datum["y_pred_upp"] - alt.datum["y_pred_low"]
        )

        circle = base.mark_circle(size=30).encode(
            x=alt.X("idx:Q", axis=alt.Axis(format=",d"), title="Ordered Samples"),
            y=alt.Y(
                "y_true:Q",
                axis=alt.Axis(format="$,d"),
                title="Observed Values and Prediction Intervals (centered)",
            ),
            color=alt.value("#f2a619"),
            tooltip=tooltip,
        )

        bar = base.mark_bar(opacity=0.8, width=2).encode(
            x=alt.X("idx:Q", scale=alt.Scale(domain=x_domain, padding=0), title=""),
            y=alt.Y("y_pred_low:Q", scale=alt.Scale(domain=y_domain, padding=0), title=""),
            y2=alt.Y2("y_pred_upp:Q", title=None),
            color=alt.value("#e0f2ff"),
            tooltip=tooltip,
        )

        tick = base.mark_tick(opacity=0.4, orient="horizontal", thickness=1, width=5).encode(
            x=alt.X("idx:Q", title=""),
            color=alt.value("#006aff"),
        )
        tick_low = tick.encode(y=alt.Y("y_pred_low:Q", title=""))
        tick_upp = tick.encode(y=alt.Y("y_pred_upp:Q", title=""))

        chart = bar + tick_low + tick_upp + circle
        return chart

    chart1 = plot_calibration(df).properties(height=250, width=325)
    chart2 = plot_intervals(df).properties(height=250, width=325)
    chart = chart1 | chart2

    return chart

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

df = Random_Forest_QuantileRegressor(X,Y,5)
chart = plot_calibration_and_intervals(df)
chart

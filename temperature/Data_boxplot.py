
# important
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


from sklearn.preprocessing import MinMaxScaler






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
    X = np.round(X,3)
    Y = (np.array(np.round(Y,3)))
    
        
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.round(X,2)
    
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
def box_plot(data):
    fig = plt.figure(figsize =(10, 4))
    ax = fig.add_subplot(111)
 
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                notch ='True', vert = 0)
    
    colors = ['#D89800', '#00FF00', '#0C00F7','#B53222',
             '#F5FF00','#FF00FF','#FBFFAF','#00FFFF']
    
    print("00")
    for patch, color in zip(bp['boxes'], colors):
         patch.set_facecolor(color)
 
      
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
       whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
       cap.set(color ='#8B008B',
            linewidth = 2)
   
    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color ='black',
               linewidth = 3)
 
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
    
    # x-axis labels
    ax.set_yticklabels(Factors)
    
    # Adding title 
    plt.title("CFeatures box plot")
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
     
    # show plot
    plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#def main():
Factors= ["ed","frac","lpi","lsi","pland", "x", "y", "year"]   
# 1- create dataset 
##buildup =allb
configuration_mode =['allb','allv','alls'] 
X,Y, num_sample_per_years = create_dataset(configuration_mode[0])

box_plot(X)
'''for i in range(len(Factors)):
    fig3= plt.figure(figsize =(10, 4))
    plt.boxplot(X[:,i])
    plt.title("box plot " +str(Factors[i]))
    
'''
fig2 = plt.figure(figsize =(10, 4))
plt.boxplot(Y)
plt.title("Temperture box plot")


X,Y =oulier_removal(X,Y)
box_plot(X)
'''for i in range(len(Factors)):
    fig3= plt.figure(figsize =(10, 4))
    plt.boxplot(X[:,i])
    plt.title("box plot " +str(Factors[i]))
    
'''
fig2 = plt.figure(figsize =(10, 4))
plt.boxplot(Y)
plt.title("Temperture box plot")




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

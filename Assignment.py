from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def Assignment():
    path=input("enter tha path ")
    type=input("enter the type of file")

    if type == "excel":
        df = pd.read_excel(path, sheetname=0) 
    elif type == "csv":
        df = pd.read_csv(path)  
    else:
        print("Enter path excel or csv ")
          
    depvar=input("Enter the dependent variable name")
    
    data_info=df.info()
    print("Data information Completed",  data_info)
    
    data_des=df.describe()
    print("Data Describe Completed", data_des)
    
    removed=df.drop_duplicates()
    print("Removed the duplicates values Completed ", removed)
   
    y=df[depvar]
    x=df.drop([depvar],axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)
    print("Splited the data is Completed")
    
    for column in df:
        label_encoder=LabelEncoder()
        df[column]=label_encoder.fit_transform(df[column])
        print(" Succesful the Encoding variables")
        
    x=df.drop([depvar],axis=1)    
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)
        
    LR=LinearRegression()
    LR.fit(X_train, Y_train)
    print("LinearRegression Model Completed")
    
    score_LR=LR.score(X_test,Y_test)
    
    print("Evaluation Completed")
    
    classifier = RandomForestClassifier(n_estimators=15)
    classifier.fit(X_train, Y_train)
    print("Regression Model Completed")
    prediction=classifier.predict(X_train)
    accuracy=accuracy_score(Y_train,prediction)*100
    print("Regression Model Accuracy:", accuracy , " LinearRegression Model Accuracy:" ,score_LR*100,'%')


    return 
Assignment()
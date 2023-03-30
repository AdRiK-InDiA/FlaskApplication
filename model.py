import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

def insurance(age, bmi, children,smoker_no, smoker_yes):
    dataset = pd.read_csv('insurance.csv')

    from sklearn.preprocessing import OneHotEncoder

    ##create an instance of Onehotencoder
    encoder=OneHotEncoder()

    ## perform fit and transform
    encoded=encoder.fit_transform(dataset[['smoker']]).toarray()

    encoder_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out())

    dataset.drop('smoker',axis=1,inplace=True)
    dataset = pd.concat([dataset,encoder_df],axis=1)
    dataset.drop('region',axis=1,inplace=True)
    dataset.drop('sex',axis=1,inplace=True)

    X=dataset[['age', 'bmi', 'children','smoker_no', 'smoker_yes']] ##independent feature( DataFrame . not array)
    y=dataset['charges'] ## dependent feature array . 
    ## Train test split
    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=12)## 30% data for test 

    ## standardize the dataset Train independent data
    from sklearn.preprocessing import StandardScaler

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test) 

    ## Train the Linear Regression Model
    from sklearn.linear_model import LinearRegression

    regressor=LinearRegression()

    regressor.fit(X_train,y_train) ## train the model

    data = np.array([[age, bmi, children,smoker_no, smoker_yes]])

    data = scaler.transform(data)

    return regressor.predict(data)[0]
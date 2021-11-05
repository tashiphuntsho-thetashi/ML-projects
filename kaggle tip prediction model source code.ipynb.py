from os import replace
import pandas as pd 
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
rfr = RandomForestRegressor()
rfr1 = RandomForestRegressor()
lab_encoder = LabelEncoder()
df = sns.load_dataset('tips')

#preprocessing steps 
bin_category = [feature for feature in df.columns if ((str(df[feature].dtype) == 'category') and (len(df[feature].unique()) == 2))]
for feature in bin_category:
    df[feature] = lab_encoder.fit_transform(df[feature])
day = pd.get_dummies(df['day'],drop_first=True)
df.drop('day',axis =1 ,inplace = True)
df1 = pd.concat([df,day],axis =1)

# handling outliers:
# after visualizing the dataset we have come to the conclusion that tip and total_bill columns contains some 
# outliers.  
max_tip = df1['tip'].quantile(0.95)
max_bill = df1['total_bill'].quantile(0.95)
# removing the outliers
df1 = df1[(df1['total_bill'] <= max_bill)]
df1 = df1[df1['tip'] <= max_tip]
# splitting the dataset into test and train set
X = df.drop('tip',axis =1 )
Y = df['tip']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

rfr.fit(x_train,y_train)
print("training score: {} \ntesting score:{}".format(rfr.score(x_train,y_train),rfr.score(x_test,y_test)))

# we can see that the accuracy of the model on both training and testing set is not very good. 
# so we will increase the sample size. 

# we not trying other models because for this type of regression problems which takes maximum number of categorical
# features as inputs random forest will perform the best as it takes decision trees as the base estimators.

df2 = df1.sample(n = 1000, replace = True)
# splitting the dataset again and then training the model
X1 , Y1 = df2.drop('tip',axis =1) , df2['tip']
x1_train,x1_test,y1_train,y1_test = train_test_split(X1,Y1,test_size = 0.2, random_state=0)

rfr1.fit(x1_train,y1_train)
print("accuracy on train data : {}".format(rfr1.score(x1_train,y1_train)))
print("accuracy on test data : {}".format(rfr1.score(x1_test,y1_test)))

filename = 'tip_prediction_model.sav'

pickle.dump(rfr1,open(filename,'wb'))



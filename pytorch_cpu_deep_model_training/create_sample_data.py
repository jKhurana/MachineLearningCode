import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

total_data_points = 1000
feature_dim = 40

feature_name = ["f_"+str(i) for i in range(feature_dim)]

x_data = np.random.normal(0,1,size=(total_data_points,feature_dim))
y_data = np.random.randint(0,2,size=(total_data_points,))

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# create pandas data frame from numpy array for training data
train_df = pd.DataFrame(X_train)
train_df.columns = feature_name
train_df['target'] = y_train

# create pandas data frame from numpy array for test data
test_df = pd.DataFrame(X_test)
test_df.columns = feature_name
test_df['target'] = y_test

train_df.to_csv('./data/train.csv',index=False)
test_df.to_csv('./data/test.csv',index=False)
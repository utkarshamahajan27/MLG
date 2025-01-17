# -*- coding: utf-8 -*-
"""LaptopPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uim4dc21uxLeblUm6upg776sCLTTpm7G
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./drive/MyDrive/laptop_data.csv')

df.head()

df.shape

df.info()

df.duplicated().sum() #to check for duplicates

df.isnull().sum() #to check for missing values

df.drop(columns=['Unnamed: 0'], inplace=True)

df.head()

df['Ram']= df['Ram'].str.replace('GB','')
df['Weight']= df['Weight'].str.replace('kg','')

df.head()

df['Ram']= df['Ram'].astype('int32')
df['Weight']= df['Weight'].astype('float32')

df.info()

import seaborn as sns
sns.distplot(df['Price']) #to check price distrbution of the laptops

df['Company'].value_counts().plot(kind='bar')

sns.barplot(x=df['Company'], y= df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['TypeName'].value_counts().plot(kind='bar')

sns.barplot(x=df['TypeName'], y= df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['Inches'])

sns.scatterplot(x=df['Inches'], y=df['Price']) #not a strong correlation with price

df['ScreenResolution'].value_counts()

df['Touchscreen']= df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
df.head()

df['Touchscreen'].value_counts().plot(kind='bar')

sns.barplot(x=df['Touchscreen'], y=df['Price'])

df['Ips']= df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.head()

sns.barplot(x=df['Ips'], y=df['Price'])

new = df['ScreenResolution'].str.split('x',n=1, expand=True)

df['X_res']= new[0]
df['Y_res']= new[1]
df.head()

df['X_res']= df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df.head()

df['X_res']= df['X_res'].astype('int32')
df['Y_res']= df['Y_res'].astype('int32')

df.info()

numeric_df = df.select_dtypes(include=['float64', 'float32', 'int32', 'int64'])
numeric_df.corr()

numeric_df.corr()['Price']

df['ppi']= (((df['X_res']**2 ) + (df['Y_res']**2)) ** 0.5/df['Inches']).astype('float')

numeric_df = df.select_dtypes(include=['float64', 'float32', 'int32', 'int64'])
numeric_df.corr()
numeric_df.corr()['Price']

df.drop(columns=['ScreenResolution'], inplace=True)

df.head()

df.drop(columns=['Inches','X_res','Y_res'], inplace=True)

df.head()

df['Cpu'].value_counts()

df['Cpu Name']= df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

df.head()

def fetch_processor(text):
  if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
    return text

  else:
    if text.split()[0] == 'Intel':
      return 'Other Intel Processor'

    else:
      return 'AMD Processor'

df['Cpu brand']= df['Cpu Name'].apply(fetch_processor)

df.head()

df['Cpu brand'].value_counts().plot(kind='bar')

sns.barplot(x= df['Cpu brand'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns=['Cpu','Cpu Name'], inplace=True)

df.head()

df['Ram'].value_counts().plot(kind='bar')

sns.barplot(x= df['Ram'], y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['Memory'].value_counts()

import pandas as pd

# Assuming 'df' is your DataFrame
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = new[0].str.strip()
df["second"] = new[1].str.strip() if new.shape[1] > 1 else None

# Apply storage type checks
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Remove non-numeric characters
df['first'] = df['first'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)

df["second"].fillna("0", inplace=True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

# Remove non-numeric characters
df['second'] = df['second'].str.extract(r'(\d+)', expand=False).fillna(0).astype(int)

# Compute storage capacities
df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
df["Hybrid"] = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]

# Drop temporary columns
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                 'Layer2Flash_Storage'], inplace=True)

df.drop(columns=['Memory'], inplace=True)

df.head()

numeric_df = df.select_dtypes(include=['float64', 'float32', 'int32', 'int64'])
numeric_df.corr()
numeric_df.corr()['Price']

df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

df.head()

df['Gpu'].value_counts()

df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])

df.head()

df['Gpu brand'].value_counts()

df = df[df['Gpu brand'] != 'ARM']

df['Gpu brand'].value_counts()

sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns=['Gpu'],inplace=True)

df.head()

df['OpSys'].value_counts()

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)

df.head()

df.drop(columns=['OpSys'],inplace=True)

sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['Weight'])

sns.scatterplot(x=df['Weight'],y=df['Price'])

numeric_df = df.select_dtypes(include=['float64', 'float32', 'int32', 'int64'])
numeric_df.corr()
numeric_df.corr()['Price']

sns.heatmap(numeric_df.corr())

sns.distplot(np.log(df['Price']))

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

X_train

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

pip install graphviz pydotplus

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

# Extract the trained decision tree
trained_tree = pipe.named_steps['step2']

pip install pydotplus

from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
feature_names = pipe.named_steps['step1'].get_feature_names_out()
dot_data = export_graphviz(trained_tree, out_file=None,
                           feature_names=feature_names,
                           filled=True, rounded=True,
                           special_characters=True)

# Create a graph from dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Display the tree
Image(graph.create_png())

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(trained_tree, feature_names=feature_names, filled=True, rounded=True)
plt.show()

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))

from sklearn.ensemble import VotingRegressor


# Define the ColumnTransformer for the preprocessing step
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

# Define the individual models
rf = RandomForestRegressor(n_estimators=350, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)
dr = DecisionTreeRegressor(max_depth=8)
lr= LinearRegression()

# Define the VotingRegressor
voting_regressor = VotingRegressor([('rf', rf), ('dr', dr), ('lr', lr)], weights=[1,1,1])

pipe = Pipeline([
        ('step1', step1),
        ('step2', voting_regressor)
    ])

pipe= pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f'R2 score: {r2_score(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')

import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))

df
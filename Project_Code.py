#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans


df = pd.read_csv(r"D:\Document\MCA3\Python\activity3\africa_individual_migration.csv")

df.head(50)


# In[32]:


df.tail()


# In[33]:


# Print Age of all migrants from "Chad"
print(df.loc[df['Country_of_Birth'] == 'Chad'])


# In[8]:


# Basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check unique values in categorical columns
print(df.nunique())

# Quick statistics
print(df.describe())


# In[9]:


#Handle Missing Values
# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# In[11]:


#Exploratory Data Analysis:
# 1. Gender Distribution
sns.countplot(data=df, x='Gender')
plt.title("Gender Distribution of Migrants")
plt.show()


# In[13]:


#2. Age Distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution of Migrants")
plt.show()


# In[15]:


#3. Migration Reason
df['Migration_Reason'].value_counts().plot(kind='bar')
plt.title("Migration Reason Distribution")
plt.show()


# In[17]:


# 4. Education Level
sns.countplot(data=df, x='Education_Level')
plt.title("Education Level of Migrants")
plt.xticks(rotation=45)
plt.show()


# In[19]:


#5. Year of Migration Trend
df.groupby('Year_of_Migration').size().plot()
plt.title("Number of Migrants Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Migrants")
plt.show()


# In[2]:


# 6. Origin vs Destination Heatmap
pivot = df.pivot_table(index='Country_of_Birth', 
                       columns='Destination_Country', 
                       aggfunc='size', 
                       fill_value=0)

plt.figure(figsize=(14,10))
sns.heatmap(pivot, cmap="viridis")
plt.title("Migration Flow: Origin → Destination")
plt.show()


# In[5]:


# 1. Select features & label
X = df[['Age', 'Gender', 'Country_of_Birth', 
        'Destination_Country', 'Education_Level']]
y = df['Migration_Reason']   # Target variable


# In[6]:


# 2. Encode categorical data
label = LabelEncoder()

df['Gender'] = label.fit_transform(df['Gender'])
df['Country_of_Birth'] = label.fit_transform(df['Country_of_Birth'])
df['Destination_Country'] = label.fit_transform(df['Destination_Country'])
df['Education_Level'] = label.fit_transform(df['Education_Level'])
df['Migration_Reason'] = label.fit_transform(df['Migration_Reason'])

X = df[['Age', 'Gender', 'Country_of_Birth', 
        'Destination_Country', 'Education_Level']]
y = df['Migration_Reason']


# In[8]:


# 3. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[10]:


# 5. Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[12]:


#CLUSTERING
df_cluster = df[['Age', 'Gender', 'Education_Level']]

# Encode education level
df_cluster = df_cluster.copy()
df_cluster['Education_Level'] = label.fit_transform(df_cluster['Education_Level'])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)


# In[15]:


# KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(df[['Age', 'Gender', 'Education_Level', 'Cluster']].head())


# In[16]:


#REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[['Age', 'Gender']]
y = df['Year_of_Migration']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))


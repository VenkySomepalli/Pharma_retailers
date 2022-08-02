# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:29:34 2022

@author: venki
"""
######## Phrama Retailer Data set  provided by 360DigiTMG ##############

import pandas as pd ## data manipulation
import numpy as np
# Load the data set

retail = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\Dataset-2022_07_13\\Dataset\\CreditAnalysis_data.csv")
retail.head()
retail.info()  # information about the data null values, type and memory
retail.describe() ## stastical information
retail.shape  # (19148, 15)
retail.columns
"""['Unnamed: 0', 'master_order_id', 'master_order_status', 'created',
       'order_id', 'order_status', 'ordereditem_quantity', 'prod_names',
       'ordereditem_unit_price_net', 'ordereditem_product_id', 'value',
       'group', 'dist_names', 'retailer_names', 'bill_amount']"""

retail.drop(['Unnamed: 0'], axis = 1, inplace = True)
retail.shape
# EDA and Data Preprocessing
retail.dtypes

retail.duplicated().sum()  # no duplicates

retail.isna().sum()

from sklearn.impute import SimpleImputer
#### Mean Imputer
mean_imputer = SimpleImputer(missing_values = np.nan , strategy='mean')
retail["ordereditem_product_id"] = pd.DataFrame(mean_imputer.fit_transform(retail[["ordereditem_product_id"]]))
retail["ordereditem_product_id"].isna().sum()

retail.isna().sum()  ## no null values

cols = ['master_order_id','order_id', 'ordereditem_quantity', 
       'ordereditem_unit_price_net', 'ordereditem_product_id', 'value', 'bill_amount']
import matplotlib.pyplot as plt
import seaborn as sns

# Check outliers
for i in cols:
    sns.boxplot(retail[i]); plt.show() 
#Out lier treatment with winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method ='iqr', # choose IQR rule boundaries or gaussian for mean and std
                   tail = 'both', # cap left, right or both tails
                   fold = 1.5,
                  # variables = ['']
                  )

for i in cols:
    retail[i] = winsor.fit_transform(retail[[i]])

for i in cols:
    sns.boxplot(retail[i]); plt.show()  # no outliers
bx = sns.boxplot(data = retail, orient ="h", palette = "Set2")
retail.var()  ## no zero variance
# check normal distribution or not
sns.displot(retail.value, kde = True) #Right skewed
sns.displot(np.sqrt(retail.value), kde = True)  

# Recency: How recently has the retailer made a transaction.

## Calculating Recency

recency = pd.DataFrame(retail.groupby('retailer_names')['created'].max().reset_index())
recency['created'] = pd.to_datetime(recency['created']).dt.date
recency['MaxDate'] = recency['created'].max()
recency['recency'] = (recency['MaxDate'] - recency['created']).dt.days + 1
recency = recency[['retailer_names','recency']]
recency.head()

# Frequency: How frequent is the retailer in buying some product.

## Calculating Recency

frequency = pd.DataFrame(retail.groupby('retailer_names')['master_order_id'].nunique().reset_index())
frequency.columns = ['F_retailer_names','frequency']
frequency.head()

# Monetary: How much does the retailer pay on purchasing product.
# Calculating Monetary

monetary = pd.DataFrame(retail.groupby('retailer_names')['value'].sum().reset_index())
monetary.columns = ['M_retailer_names','monetary']
monetary.head()


## All Recency, Frequency and Monetary with retailer_names.

# combining the three into one table

rfm = pd.concat([recency,frequency,monetary], axis=1)
rfm
rfm.drop(['F_retailer_names','M_retailer_names'], axis=1, inplace=True)
rfm.head(10)
rfm.shape ## (215, 4)
print(rfm)

#Ranking retailer’s based upon their recency, frequency, and monetary score

rfm['R_rank'] = rfm['recency'].rank(ascending=False)
rfm['F_rank'] = rfm['frequency'].rank(ascending=True)
rfm['M_rank'] = rfm['monetary'].rank(ascending=True)

# normalizing the rank of the customers
rfm['R_rank_norm'] = (rfm['R_rank']/rfm['R_rank'].max())*100
rfm['F_rank_norm'] = (rfm['F_rank']/rfm['F_rank'].max())*100
rfm['M_rank_norm'] = (rfm['F_rank']/rfm['M_rank'].max())*100
 
rfm.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
 
rfm.head()

## Calculating RFM score.

rfm['RFM_Score'] = 0.15*rfm['R_rank_norm'] + 0.28*rfm['F_rank_norm'] + 0.57*rfm['M_rank_norm']
rfm = rfm.round(0)
rfm.head()

rfm['RFM_Score'].value_counts()


## Retailer type

rfm["Retailer_segment"] = np.where(rfm['RFM_Score'] >
                                      90, "Top retailers",
                                      (np.where(
                                        rfm['RFM_Score'] > 70,
                                        "High value retailers",
                                        (np.where(
    rfm['RFM_Score'] > 50,
                             "Medium Value retailers",
                             np.where(rfm['RFM_Score'] > 25,
                            'Low Value retailers', 'Lost retailers'))))))
rfm.head()
rfm['Retailer_segment'].value_counts()

plt.pie(rfm.Retailer_segment.value_counts(),
        labels=rfm.Retailer_segment.value_counts().index,
        autopct='%.0f%%')
plt.show()

# looking the RFM value for each of the category
rfm.groupby('Retailer_segment')['recency','frequency','monetary'].mean().round(0)


column = ['recency','frequency','monetary']
plt.figure(figsize=(15,4))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    rfm.groupby('Retailer_segment')[j].mean().round(0).plot(kind='bar', color='blue')
    plt.title('What is the {} of each customer type'.format(j), size=13)
    plt.xlabel('')
    plt.xticks(rotation=45)

plt.show()

rfm.shape # (215, 9)

rfm.to_csv("rfm.csv", encoding = "utf-8")
##save file
import os
os.getcwd()

########### K-Means clustering. ###############################

data_km = rfm.copy()
data_km = data_km.iloc[:, :4]
data_km.columns

#or 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_km = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\rfm.csv")
data_km = data_km.iloc[:, :5]
data_km.columns
data_km.drop(['Unnamed: 0'], axis = 1, inplace = True)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(data_km.recency, data_km.frequency, color='grey', alpha=0.3)
plt.title('Recency vs Frequency', size=15)
plt.subplot(1,3,2)
plt.scatter(data_km.monetary, data_km.frequency, color='grey', alpha=0.3)
plt.title('Monetary vs Frequency', size=15)
plt.subplot(1,3,3)
plt.scatter(data_km.recency, data_km.monetary, color='grey', alpha=0.3)
plt.title('Recency vs Monetary', size=15)
plt.show()

column = ['recency', 'frequency', 'monetary']
for i in column:
    sns.boxplot(data_km[i]); plt.show() 

#Out lier treatment with winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method ='iqr', # choose IQR rule boundaries or gaussian for mean and std
                   tail = 'both', # cap left, right or both tails
                   fold = 1.5,
                  # variables = ['']
                  )

for i in column:
    data_km[i] = winsor.fit_transform(data_km[[i]])

for i in column:
    sns.boxplot(data_km[i]); plt.show()  # no outliers

cx = sns.boxplot(data = data_km,orient ="h", palette = "Set2" )
# removing customer id as it will not used in making cluster
data_km = data_km.iloc[:,1:]
data_km.columns

## Normalization or min max scaling
def norm_fun(i):
    x = (i - i.max())/(i.max() - i.min())
    return (x)

df_norm = norm_fun(data_km)
data_norm = pd.DataFrame(df_norm)
df_norm.describe()

###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []  ## Total with sum of the squares.
k = list(range(2, 11))  ## range of clusters, k=2,3,4,5,6,7,8,9,10,11
## if use more than 10, its a meaning less because of we have only 25 universities.
## Think logically and decide how many clusters.
#for each value of i, k means clustering is done and  inertia is recorded
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_norm)
    TWSS.append(kmeans.inertia_)  ## data filled in TWSS
    
TWSS ## for each of the k value, for 9 clusters have less records in each cluster
# and then get less WSS.

# Scree plot/elbow curve
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

## finding Silhoutte score 
from sklearn.metrics import silhouette_score
silhouette_avg_norm = []
for num_clusters in k:
    # initialise kmeans
    kmeans1 = KMeans(n_clusters =num_clusters)
    kmeans1.fit(data_norm)
    cluster_labels = kmeans1.labels_
    
    # silhouette score
    silhouette_avg_norm.append(silhouette_score(data_norm, cluster_labels))

plt.plot(k, silhouette_avg_norm, 'bx-') 
plt.xlabel('values of k') 
plt.ylabel ('silhouette score') 
plt.show()

## Mini Max scalar(normalization)  from silhoutte score we got k = 4


## Standardization
from sklearn.preprocessing import StandardScaler
# scaling the variables and store it in different data
standard_scaler = StandardScaler()
data_sc = standard_scaler.fit_transform(data_km)

# converting it into dataframe
data_km_norm = pd.DataFrame(data_sc)
data_km_norm.columns = ['recency','frequency','monetary']
data_km_norm.head()

###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []  ## Total with sum of the squares.
k = list(range(2, 11))  ## range of clusters, k=2,3,4,5,6,7,8,9,10,11
## if use more than 10, its a meaning less because of we have only 25 universities.
## Think logically and decide how many clusters.
#for each value of i, k means clustering is done and  inertia is recorded
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_km_norm)
    TWSS.append(kmeans.inertia_)  ## data filled in TWSS
    
TWSS ## for each of the k value, for 9 clusters have less records in each cluster
# and then get less WSS.

# Scree plot/elbow curve
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

## finding Silhoutte score 
from sklearn.metrics import silhouette_score
silhouette_avg_norm = []
for num_clusters in k:
    # initialise kmeans
    kmeans1 = KMeans(n_clusters =num_clusters)
    kmeans1.fit(data_km_norm)
    cluster_labels = kmeans1.labels_
    
    # silhouette score
    silhouette_avg_norm.append(silhouette_score(data_km_norm, cluster_labels))

plt.plot(k, silhouette_avg_norm, 'bx-') 
plt.xlabel('values of k') 
plt.ylabel ('silhouette score') 
plt.show()

## Standard scalar silhoute_score conforms a k = 3

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)  # check n_clusters is 3
y = model.fit(data_km_norm)


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_km['clust'] = mb # creating a  new column and assigning it to new column 

data_km.head()
data_km_norm.head()
## Visulaizing clusters

data_km = data_km.iloc[:,[3,0,1,2]] ## rearranging the columns
data_km.head()

data_km.iloc[:, 0:4].groupby(mb).mean() ## [rows; coulmn:coulmn]

data_km.to_csv("retail_km_3.csv", encoding = "utf-8")
import os
os.getcwd()


## Normalization silhoute_score conforms a k = 4

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)  # check n_clusters is 4
y = model.fit(data_norm)


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_km['clust'] = mb # creating a  new column and assigning it to new column 

data_km.head()
data_norm.head()
## Visulaizing clusters

data_km = data_km.iloc[:,[3,0,1,2]] ## rearranging the columns
data_km.head()

data_km.iloc[:, 0:4].groupby(mb).mean() ## [rows; coulmn:coulmn]

data_km.to_csv("retail_km_4.csv", encoding = "utf-8")
import os
os.getcwd()

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)  # check n_clusters = 5
y = model.fit(data_km_norm)


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_km['clust'] = mb # creating a  new column and assigning it to new column 

data_km.head()
data_km_norm.head()
## Visulaizing clusters

data_km = data_km.iloc[:,[1,0,2,3]] ## rearranging the columns
data_km.head()

data_km.iloc[:, 0:4].groupby(mb).mean() ## [rows; coulmn:coulmn]

data_km.to_csv("retail_km_5.csv", encoding = "utf-8")
##save file
import os
os.getcwd()

column = ['recency','frequency','monetary']
plt.figure(figsize=(15,4))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    sns.boxplot(y = data_km[j], x = data_km['clust'], palette = 'spring')
    plt.title('{} wrt clusters'.format(j.upper()), size=13)
    plt.ylabel('')
    plt.xlabel('')

plt.show()

# Creating figure
fig = plt.figure(figsize = (8, 5))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(data_km.recency, data_km.frequency, data_km.monetary, c = data_km.clust, cmap='Accent')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('RFM in 3D with Clusters', size=15)
ax.set(facecolor='white')
plt.show()

data_km['clust'].value_counts()



##################### Classification on 3 clusters data ######################

data_class = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\retail_km_3.csv")

data_class.columns
# ['Unnamed: 0', 'clust', 'monetary', 'recency', 'frequency']

data_class.info()
sns.distplot(data_class['frequency'])

data_class.drop(['Unnamed: 0'], axis = 1, inplace = True)
X = data_class.drop('clust', axis = 1)
Y = data_class.clust
data_class.describe()

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", random_state=0,max_depth=5, min_samples_leaf=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Accuracy =  0.9846153846153847
y_pred_train = clf.predict(X_train)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
# accuracy = 0.9933333333333333

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
clust_3 = accuracy_score(y_test, y_pred )

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, y_pred_train)}")
print(f"Test Accuracy of Decision Tree Classifier is {clust_3} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = ['monetary', 'recency', 'frequency']
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

########## or use below one for tree image #######

from sklearn.tree import plot_tree
plt.figure(figsize = (15,10))
plot_tree(clf, filled = True, feature_names = feature_cols)
plt.show()

df_importance = pd.DataFrame({'Features' : feature_cols, 'Importances': clf.feature_importances_})
df_importance

import pickle
pickle.dump(clf, open('clust_3.pkl','wb'))

## Checking the result of pickle file.
model = pickle.load(open("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\clust_3.pkl", "rb"))
result = model.score(X_test, y_test)
print(result)

##################### Classification on 4 clusters data ######################

data_class = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\retail_km_4.csv")
data_class.shape
data_class.columns
data_class['frequency'].value_counts()
# ['Unnamed: 0', 'clust', 'monetary', 'recency', 'frequency']
data_class.drop(['Unnamed: 0'], axis = 1, inplace = True)
X = data_class.drop('clust', axis = 1)
Y = data_class.clust

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=28) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", random_state=28,max_depth=5, min_samples_leaf=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Accuracy =  0.9692307692307692
y_pred_train = clf.predict(X_train)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
# accuracy = 0.9933333333333333

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
clust_4 = accuracy_score(y_test, y_pred )

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, y_pred_train)}")
print(f"Test Accuracy of Decision Tree Classifier is {clust_4} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")



from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = ['monetary', 'recency', 'frequency']
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

########## or use below one for tree image #######

from sklearn.tree import plot_tree
plt.figure(figsize = (15,10))
plot_tree(clf, filled = True)
plt.show()



##################### Classification on 5 clusters data ######################

data_class = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\DS_Project_Team_72\\DS_Project_Team_72\\retail_km_5.csv")

data_class.columns
# ['Unnamed: 0', 'clust', 'monetary', 'recency', 'frequency']
data_class.drop(['Unnamed: 0'], axis = 1, inplace = True)
X = data_class.drop('clust', axis = 1)
Y = data_class.clust

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=5, min_samples_leaf=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Accuracy = 0.9692307692307692
y_pred_train = clf.predict(X_train)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
## Accuracy = 0.9733333333333334

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
clust_5 = accuracy_score(y_test, y_pred )

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, y_pred_train)}")
print(f"Test Accuracy of Decision Tree Classifier is {clust_5} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")


from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = ['monetary', 'recency', 'frequency']
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

########## or use below one for tree image #######

from sklearn.tree import plot_tree
plt.figure(figsize = (15,10))
plot_tree(clf, filled = True)
plt.show()


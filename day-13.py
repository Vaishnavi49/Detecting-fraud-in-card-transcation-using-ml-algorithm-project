#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
data = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
data


# ## 2a. BASIC EDA

# In[69]:


data.shape


# In[70]:


data.size


# In[71]:


data.ndim


# In[72]:


data.info()


# In[73]:


data.describe()


# In[74]:


data.columns[data.isna().any()]


# In[75]:


data.columns


# In[76]:


data["fraud"].unique()


# ## 2b. ADVANCE EDA

# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x='distance_from_home',y='distance_from_last_transaction',data=data)#ploting
plt.title("distance_from_home v/s distance_from_last_transaction plot using seaborn")
plt.grid() #displaying 


# In[78]:


sns.stripplot(x='repeat_retailer',y='used_chip',data=data)#ploting


# In[79]:


sns.violinplot(x='repeat_retailer',y='used_chip',data=data)#ploting


# In[80]:


sns.boxplot(x='repeat_retailer',y='used_chip',data=data)#ploting


# In[81]:


sns.histplot(x='repeat_retailer',y='used_chip',data=data)#ploting


# In[82]:


sns.displot(data['fraud'])


# In[83]:


sns.barplot(x='repeat_retailer',y='used_chip',data=data)#ploting


# In[84]:


sns.countplot(x='used_chip',hue='fraud',data=data)


# In[85]:


#heat map
import seaborn as sns
import matplotlib.pyplot as plt

data["fraud"]=data["fraud"].map(
{0.0:1, 1.0:2})
sns.heatmap(data.corr()) #df.corr()gives us correlation matrix for df


# ## 2c. FEATURE ENGINEERING AND FEATURE SELECTION

# In[136]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
data_pda = data_pda.dropna()# droping NaN

x_columns =['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']
data_pda["fraud"]=data_pda["fraud"].map(
{0:1, 1:2})
#segregation of input and output column
x = data_pda.iloc[:,:3].values
y = data_pda.iloc[:,-1].values

#splitting data into trining and testing partion
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



# In[137]:


data_pda


# ## 3. MODEL TRAINING

# In[138]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
from sklearn.linear_model import LogisticRegression # importing algo
Logistic_regression_model = LogisticRegression() # initialising the algo
Logistic_regression_model.fit(x_train,y_train) # traing the algo on traing partitions

print("[INFO] model training complete..")


# ## 4. MODEL EVALUATION

# In[139]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
#making the model predict output for x_test
y_predicted = Logistic_regression_model.predict(x_test)

#taking actual value
y_actual = y_test


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import math

accuracy = accuracy_score(y_actual,y_predicted)
print(f"accuracy of given model is:{accuracy}")

precision = precision_score(y_actual,y_predicted)
print(f"precision of given model is:{precision}")

recall = recall_score(y_actual,y_predicted)
print(f"recall of given model is:{recall}")

f1 = f1_score(y_actual,y_predicted)
print(f"f1 of given model is:{f1}")



# In[98]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
#USING THE MODEL TO PREDICT OUTPUT FOR NEW INPUT

distance_from_home = float(input("enter the distance_from_home :"))
distance_from_last_transaction = float(input("enter the distance_from_last_transaction :"))
ratio_to_median_purchase_price = float(input("enter ratio_to_median_purchase_price :"))




#storing the user input in a 2-D array
new_user_input = [[distance_from_home,distance_from_last_transaction,ratio_to_median_purchase_price]]

#using model to pridect the output for new input
new_output = Logistic_regression_model.predict(new_user_input)

#output is a number, we will give answer in text
if new_output[0] ==0:
    print("fraud is 0")
if new_output[0] ==1:
    print("fraud is 1")


# # STANDRAD SCALING

# In[131]:


import seaborn as sns 
import sklearn
import pandas as pd
data = data.dropna()# droping NaN
#segregation of input and output column
x = data.iloc[:,:3].values
y = data.iloc[:,-1].values

#splitting data into trining and testing partion
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



from sklearn.preprocessing import StandardScaler
sc =StandardScaler() #initiaslise the standrad
x_scale_standrad_scaled = sc.fit_transform(x_train)
x_test_standrad_scaled = sc.fit(x_test)
print(x_scale_standrad_scaled)
print()
print(x_test_standrad_scaled)


# ## 3. MODEL TRAINING

# In[132]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
from sklearn.linear_model import LogisticRegression # importing algo
Logistic_regression_model = LogisticRegression() # initialising the algo
Logistic_regression_model.fit(x_train,y_train) # traing the algo on traing partitions

print("[INFO] model training complete..")


# ## 4. MODEL EVALUATION

# In[133]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
#making the model predict output for x_test
y_predicted = Logistic_regression_model.predict(x_test)

#taking actual value
y_actual = y_test


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import math

accuracy = accuracy_score(y_actual,y_predicted)
print(f"accuracy of given model is:{accuracy}")

precision = precision_score(y_actual,y_predicted)
print(f"precision of given model is:{precision}")

recall = recall_score(y_actual,y_predicted)
print(f"recall of given model is:{recall}")

f1 = f1_score(y_actual,y_predicted)
print(f"f1 of given model is:{f1}")



# # MIN_MAX SCALING

# In[120]:


x = data.iloc[:,:3].values #selecting features /input columns
y = data.iloc[:,-1].values #selecting target/output columns
from sklearn.preprocessing import MinMaxScaler #importing Min - Max scaling
min_max =MinMaxScaler() #initiaslise the MinMax Scaler

x_min_max_scaled = min_max.fit_transform(x)#using min_max scaler on x
x_min_max_scaled_df = pd.DataFrame(x_min_max_scaled)# converting back to dataframe
print(x_min_max_scaled)
print(x_min_max_scaled_df)


# ## 3. MODEL TRAINING

# In[121]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
from sklearn.linear_model import LogisticRegression # importing algo
Logistic_regression_model = LogisticRegression() # initialising the algo
Logistic_regression_model.fit(x_train,y_train) # traing the algo on traing partitions

print("[INFO] model training complete..")


# ## 4. MODEL EVALUATION

# In[130]:


data_pda = pd.read_csv("/Users/vaishnavikhot/Downloads/card_transdata.csv")
#making the model predict output for x_test
y_predicted = Logistic_regression_model.predict(x_test)

#taking actual value
y_actual = y_test


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import math

accuracy = accuracy_score(y_actual,y_predicted)
print(f"accuracy of given model is:{accuracy}")

precision = precision_score(y_actual,y_predicted)
print(f"precision of given model is:{precision}")

recall = recall_score(y_actual,y_predicted)
print(f"recall of given model is:{recall}")

f1 = f1_score(y_actual,y_predicted)
print(f"f1 of given model is:{f1}")



# ### BAR PLOT

# In[134]:


#accuracy
labels = ['without scaling',"min_max","standard"]
accuracy = [92.106,92.1865,92.129]
import seaborn as sns
sns.barplot(x= labels,y = accuracy)


# In[140]:


#precision
labels = ['without scaling',"min_max","standard"]
precision = [93.37917475046422,62.35613604034657,61.7144325922118]
import seaborn as sns
sns.barplot(x= labels,y = precision)


# In[141]:


#recall
labels = ['without scaling',"min_max","standard"]
recall = [98.38336676202063,27.49458319078572,27.33845715912326]
import seaborn as sns
sns.barplot(x= labels,y = recall)


# In[142]:


#f1
labels = ['without scaling',"min_max","standard"]
f1 = [95.81597662348059,38.16232044636144,37.89158052552672]
import seaborn as sns
sns.barplot(x= labels,y = f1)


# In[ ]:





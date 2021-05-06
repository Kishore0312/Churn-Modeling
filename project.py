import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

#--DATASET DECRIPTION--#
data=pd.read_csv("c:/Users/User/Desktop/churn.csv")
print(data.head(5))
print(data.describe())
print(data.info())
print(data.isnull().sum())

#--EDA--#
gender_count=data['Gender'].value_counts()
#print(gender_count)
    #--PLOTTING THE FEATURE OF DATA TO FIND CORRELATION BETWEEN THEM--#
gender=data["Gender"]
sns.countplot(gender)
plt.title('comparison of male and female')
plt.xlabel('Gender')
plt.ylabel('population')
plt.show()
age=data['Age'].value_counts()
#print(age)
    #--COMAPRISON OF AGE--#
plt.hist(x = data.Age, bins = 10, color = 'orange')
plt.title('comparison of Age')
plt.xlabel('Age')
plt.ylabel('population')
plt.show()
geography=data['Geography'].value_counts()
print(geography)
    #--COMAPRISON OF GEOGRAPHY--#
plt.hist(x = data.Geography, bins = 5, color = 'green')
plt.title("COMAPARING THE GEOGRAPHY")
plt.xlabel("GEOGRAPHY")
plt.ylabel("POPULATION")
plt.show()

credit_card=data['HasCrCard'].value_counts()
print(credit_card)

#--COMPARISON OF HOW MANY CUSTOMERS HAS CREDIT CARD--#
plt.hist(x = data.HasCrCard, bins = 3, color = 'red')
plt.title("CUSTOMERS HAVING OR NOT HAVING THE CREDIT CARD")
plt.xlabel("CUSTOMERS HAVING CR_CARD")
plt.ylabel("POPULATION")
plt.show()

active_customers=data['IsActiveMember'].value_counts()
print(active_customers)

#--ACTIVE MEMBERS OF THE BANK--#
plt.hist(x = data.IsActiveMember, bins = 3, color = 'brown')
plt.title("ACTIVE MEMBERS")
plt.xlabel("CUSTOMERS")
plt.ylabel("POPULATION")
plt.show()

#--COMPARISON OF GEOGRAPHY AND GENDER--#
Gender = pd.crosstab(data['Gender'],data['Geography'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(6, 6))
plt.title("COMPARISON OF GEOGRAPHY AND GENDER")
plt.xlabel("GENDER")
plt.ylabel("POPULATION")
plt.show()

#--COMAPRISON OF GEOGRAPHY AND CARD HOLDERS--#
HasCrCard = pd.crosstab(data['HasCrCard'], data['Geography'])
HasCrCard.div(HasCrCard.sum(1).astype(float), axis = 0).plot(kind = 'bar',stacked = True,figsize = (6, 6))
plt.title("COMPARISON OF GEOGRAPHY AND CARD HOLDERS")
plt.xlabel("CARD HOLDERS")
plt.ylabel("GEOGRAPHY")
plt.show()

#--CALCULATING THE TOTAL BALANCE OF THE BANK ACROSS REGIONS--#
total_balance_france = data.Balance[data.Geography == 'France'].sum()
total_balance_germany = data.Balance[data.Geography == 'Germany'].sum()
total_balance_spain = data.Balance[data.Geography == 'Spain'].sum()
print("Total Balance in France :",total_balance_france)
print("Total Balance in Germany :",total_balance_germany)
print("Total Balance in Spain :",total_balance_spain)

#--DATA PRE PROCESSING--#
data["Gender"]=data["Gender"].replace("Male","0")
data["Gender"]=data["Gender"].replace("Female","1")
#print(data["Geography"].unique())
data["Geography"]=data["Geography"].replace("France","0")
data["Geography"]=data["Geography"].replace("Germany","1")
data["Geography"]=data["Geography"].replace("Spain","2")
#print(data["NumOfProducts"].unique())

#--SPLITTING THE DATASET--#
x=data.iloc[:,3:12]
y=data.iloc[:,13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

#--SCALLING--#
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
x_train = pd.DataFrame(x_train)
print(x_train.head())

print('\n')
print('\n')

#--LOGISTIC REGRESSION--#
print("LOGISTIC REGRESSION")
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
 #--CROSS VALIDATING FOR LOGISTIC REGRESSION--#
print('CROSS VALIDATION SCORES = ', cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) )
print( 'MEAN =', np.mean( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
print( 'VARIANCE = ', np.std( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
classification = classification_report(y_test,y_pred)
print("CLASSIFICATION REPORT OF LOGISTIC REGRESSION \n",classification)

print('\n')

#--RANDOM FOREST--#
print("RANDOM FOREST")
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
 #--CROSS VALIDATING FOR RANDOM FOREST--#
print('CROSS VALIDATION SCORES =', cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) )
print( 'MEAN OF SCORES =', np.mean( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
print( 'VARIANCE =', np.std( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
classification = classification_report(y_test,y_pred)
print("CLASSIFICATION REPORT OF RANDOM FOREST \n",classification)

print('\n')

#--NAIVE BAYES CLASSIFIER--#
print("NAIVE BAYES CLASSIFIER")
#--FITTING GAUSSIAN MODEL--#
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
C_matrix = confusion_matrix(y_test, y_pred)
print("The confusion matrix is :\n",C_matrix)
print("the ACcuracy score is:",accuracy_score(y_test, y_pred))
 #--CROSS VALIDATING FOR NAIVE BAYES--#
print('CROSS VALIDATION SCORES =', cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) )
print( 'MEAN OF SCORES =', np.mean( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
print( 'VARIANCE =', np.std( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
classification = classification_report(y_test,y_pred)
print("CALSSIFICATION REPORT OF NAIBVE BAYES \n",classification)

print('\n')

#--DECISION TREE--#
print("DECISION TREE")
model = DecisionTreeClassifier() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuaracy :", model.score(x_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
 #--CROSS VALIDATING FOR DECISION TREE--#
print('CROSS VALIDATION SCORES =', cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) )
print( 'MEAN OF SCORES =', np.mean( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
print( 'VARIANCE =', np.std( cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
classification = classification_report(y_test,y_pred)
print("CLASSIFICATION REPORT OF DECISION TREE\n",classification)

print('\n')

#--KNN CLASSIFIER--#
print("KNN CLASSIFIER")
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy[i] = knn.score(x_train, y_train) 
    test_accuracy[i] = knn.score(x_test, y_test)
 
plt.title('KNN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
print("Accuracy of the model is :", knn.score(x_test,y_test)*100)
y_pred = knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))
classification = classification_report(y_test,y_pred)
print("CALSSIFICATION REPORT OF KNN \n",classification)
 #--CROSS VALIDATING FOR KNN--#
print('CROSS VALIDATION SCORES =', cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) )
print( 'MEAN OF SCORES =', np.mean( cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )
print( 'VARIANCE =', np.std( cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) ) )

print('\n')

#--SUPPORT VECTOR MACHINE--#
classifier=SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)

#--TUNING THE MODEL--#
params={'C':np.arange(1,5)}
tuner = RandomizedSearchCV(estimator=classifier,param_distributions=params,n_jobs=-1,scoring='precision',
 cv=None,random_state=101,return_train_score=True,)
tuner.fit(x_train,y_train)
print("Hyper Parameter Tuning Results")
print("Best Params : ",tuner.best_params_)
print("Best Score : ",tuner.best_score_)
print("Best Model : ",tuner.best_estimator_)
best_lin_svm = tuner.best_estimator_
print(best_lin_svm)
 
#--VALIDATING THE MODEL--#
y_pred = best_lin_svm.predict(x_test)
matrix1 = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX OF LINEAR: \n",matrix1)
print("ACCURACY OF THE LINEAR MODEL",accuracy_score(y_test, y_pred))
classification_report_linear=classification_report(y_test,y_pred)
print("CLASSIFICATION REPORT OF LINEAR SVM\n",classification_report_linear)




import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
import time
import csv
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#Starter code
def get_data(FILE, multilabel=None):

    if (multilabel): # for kaggle data
        data = load_svmlight_file(FILE, multilabel=True)
        return data[0].toarray()
    # for training and testing data
    data = load_svmlight_file(FILE)
    return data[0].toarray(), data[1]
#To convert predictions to kaggle
def kaggleize(predictions,file):

    if(len(predictions.shape)==1):
        predictions.shape = [predictions.shape[0],1]

    ids = 1 + np.arange(predictions.shape[0])[None].T
    kaggle_predictions = np.hstack((ids,predictions)).astype(int)
    writer = csv.writer(open(file, 'w'))
    writer.writerow(['# id','Prediction'])
    writer.writerows(kaggle_predictions)


x_train, y_train = get_data('HW2_train.txt') # Traing data
x_test, y_test = get_data('HW2_test.txt') # Test data
x_kaggle = get_data('HW2_kaggle.txt', multilabel=True) # Kaggle data

# Default KNN 
print ("KNN")
start_time = time.time()
knn = KNeighborsClassifier()#Default classifier 
knn.fit(x_train, y_train) #Training
print("Training time--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
pred = knn.predict(x_test) #Prediction
print("Prediction time--- %s seconds ---" % (time.time() - start_time))
leng=len(pred)
#Calculating accuracy
s=0
for i in range(0,leng):
    if (pred[i]==y_test[i]):
        s=s+1
print ("Accuracy" ,(s/leng)) #Accuracy
pred_k = knn.predict(x_kaggle) #Prediciton on kaggle dataset
kaggleize(pred_k,'kaggleknn.csv') #output file
print("SVM")
# SVM

start_time = time.time()
clf = svm.SVC(kernel='linear')    #linear kernel
clf.fit(x_train, y_train) #training
print("Training time--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
pred=clf.predict(x_test) #predicting on test data

print("Prediction time--- %s seconds ---" % (time.time() - start_time))
leng=len(pred)
s=0
for i in range(0,leng):
    if (pred[i]==y_test[i]):
        s=s+1
print ("Accuracy",s/leng) #Accuracy
pred_svm = knn.predict(x_kaggle) # Predict on kaggle dataset
kaggleize(pred_svm,'kagglesvm.csv')

# Random Forest
print("Random Forest')

start_time = time.time()
rfc = RandomForestClassifier() # Default Classifier

rfc.fit(x_train, y_train) # Training
print("Traing time--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
pred=rfc.predict(x_test) # Predicting on test data
print("Prediction time--- %s seconds ---" % (time.time() - start_time))
leng=len(pred)
s=0
for i in range(0,leng):
    if (pred[i]==y_test[i]):
        s=s+1
print (s/leng) #Accuracy
pred_rfc=rfc.predict(x_kaggle) # Predicting kaggle
kaggleize(pred_rfc,'kagglerfc.csv')

#Hyper Parameter tuning

print("Hyper parameter tuning")

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200,800,1500,2500],
    'max_features': ['auto','log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5) # Choosing best estimator with Cross Validation
CV_rfc.fit(x_train, y_train)
print ("best parameters",CV_rfc.best_params_)

# getting accuracies
acc=[]
for i in range(0,4):
   # print((CV_rfc.grid_scores_[i][1]))
    acc.append((CV_rfc.grid_scores_[i][1]))
#getting accuracies   
acc2=[]
for i in range(4,8):
   # print((CV_rfc.grid_scores_[i][1]))
    acc2.append((CV_rfc.grid_scores_[i][1]))
    
n_est=[200,800,1500,2500]

#plots for log2 and auto 

plt.plot(n_est, acc)
plt.xlabel('Number of Estimators ')
plt.ylabel('Classification Accuracy')
plt.title("Max_Feature= Auto")
plt.savefig('auto.png')
plt.show()

plt.plot(n_est, acc2)
plt.xlabel('Number of Estimators ')
plt.ylabel('Classification Accuracy')
plt.title("Max_Feature= Log2")
plt.savefig('sqrt.png')
plt.show()

#combined plot
plt.plot(n_est, acc)
plt.plot(n_est, acc2)
plt.xlabel('Number of Estimators ')
plt.ylabel('Classification Accuracy')
plt.legend(['auto','log2'], loc='upper left')
plt.savefig('both.png')
plt.show()

# Model based on test accuracies
d=np.array(CV_rfc.cv_results_.get('mean_test_score')).argmax()
CV_rfc.cv_results_.get('params')[d] 
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'auto' ,n_estimators=800, oob_score = True) 
rfc.fit(x_train, y_train)
pred_rfcb=rfc.predict(x_kaggle) # Predicting on kaggle data
kaggleize(pred_rfcb,'kagglebestrfc_test.csv')


#Model based on Training acccuracies

d2=np.array(CV_rfc.cv_results_.get('mean_traing_score')).argmax() # based on traing scores
CV_rfc.cv_results_.get('params')[d2]
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'auto' ,n_estimators=200, oob_score = True) 
rfc.fit(x_train, y_train)
pred_rfcb=rfc.predict(x_kaggle) # Predicting on kaggle data
kaggleize(pred_rfcb,'kagglebestrfc_train.csv')



# Predicting on the best model 
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=1000, oob_score = True) 
rfc.fit(x_train, y_train)
pred_rfcb=rfc.predict(x_kaggle) # Predicting on kaggle data
kaggleize(pred_rfcb,'kagglebestrfc.csv')

# Gradient Descent Boosting

from sklearn.ensemble import GradientBoostingClassifier
start_time = time.time()
clf = GradientBoostingClassifier(n_estimators=2500)
clf.fit(x_train,y_train)
print("Training time --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
pred=clf.predict(x_test)
leng=len(pred)
print("Prediction time--- %s seconds ---" % (time.time() - start_time))
s=0
for i in range(0,leng):
    if (pred[i]==y_test[i]):
        s=s+1
print (s/leng)

pred_gbrfc=clf.predict(x_kaggle)
kaggleize(pred_gbrfc,'kagoutgbrfc.csv')

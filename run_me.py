# Import python modules
import numpy as np
import sys
sys.path.append('/Submission')
import kaggle

# Read in train and test data

def read_data_fb():
	print('Reading facebook dataset ...')
	trainx = np.loadtxt('C:/Users/Kailash Nathan/Documents/Books/Courses/Spring18/589/HW04/Data/data.csv', delimiter=',')
	trainy = np.loadtxt('C:/Users/Kailash Nathan/Documents/Books/Courses/Spring18/589/HW04/Data/labels.csv', delimiter=',')
	testx = np.loadtxt('C:/Users/Kailash Nathan/Documents/Books/Courses/Spring18/589/HW04/Data/kaggle_data.csv',delimiter=',')

	return (trainx, trainy, testx)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()
############################################################################
trainx, trainy, testx   = read_data_fb()
print('Train=', trainx.shape)
print('Test=', testx.shape)
 #Create dummy test output values

predicted_y = np.ones(testx.shape[0]) * -1
#utput file location
file_name = '../Predictions/best.csv'
 #Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)


#  Regression trees
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
#regr_1 = DecisionTreeRegressor(max_depth=2)

#xy=regr_1.fit(trainx,trainy)
#xyp=regr_1.predict(xte)
#xyscore=compute_error(xyp,yte)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
xtr, xte, ytr, yte = train_test_split(trainx, trainy, test_size=0.20, random_state=42) #splitting the dataset 
import time
a=[3,6,9,12,15]
b=[]
times=[]
for i in a:
    dt= DecisionTreeRegressor(max_depth=i,random_state=0, criterion="mae")
    start_time = time.time()
    score=cross_val_score(dt,xtr,ytr,cv=5)
    times.append((time.time() - start_time))
    b.append(np.mean(score)) #mean of the Cv score 
      

for i in range(len(times)): #second to milliseconds 
    times[i]=times[i]*1000

import matplotlib.pyplot as plt
plt.plot(times,b) 
plt.xlabel("Time in milliseconds")   
plt.ylabel("Out of Sample error")
plt.savefig("Q1a.jpg")
#best model fit 
dt= DecisionTreeRegressor(max_depth=15,random_state=0, criterion="mae")
x=dt.fit(xtr,ytr)
y=x.predict(xte)
print ("test error for Regression Trees",compute_error(y,yte) )#   Test error
# best model prediction on kaggle
xkg=dt.fit(trainx,trainy)
ykg=xkg.predict(testx)
ykgout=kaggle.kaggleize(ykg,'tree')
############################################################################
# Nearest Neighbour  
from sklearn.neighbors import KNeighborsRegressor
a=[3,5,10,20,25]
b=[]
times=[]
for i in a:
    neigh = KNeighborsRegressor(n_neighbors=i)
    start_time = time.time()
    score=cross_val_score(neigh,xtr,ytr,cv=5)
    times.append((time.time() - start_time))
    b.append(np.mean(score))
plt.plot(a,b) 
plt.xlabel("No of Neighbours")   
plt.ylabel("Out of Sample error")
plt.savefig("Q2a.jpg")       
#Fitting the best model
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(xtr,ytr)
y=neigh.predict(xte)
print("test error with Euclid distance",compute_error(y,yte))

neigh.fit(trainx,trainy)
knnt=neigh.predict(testx)
knnkout=kaggle.kaggleize(knnt,'kaggleknnout')
neigh = KNeighborsRegressor(n_neighbors=3,p=1,metric='manhattan')
neigh.fit(xtr,ytr)
y=neigh.predict(xte) 
print("test error with manhattan distance",compute_error(y,yte))
neigh = KNeighborsRegressor(n_neighbors=3,metric='minkowski',p=3)
neigh.fit(xtr,ytr)
y=neigh.predict(xte)
print("Test error with minkowski distance",compute_error(y,yte))

######################################################################
#Linear Models
from sklearn import linear_model
b=[]
a=[1/10**6,1/10**4,1/10**2,1,10]
#Lasso
for i in a:
    clf = linear_model.Lasso(alpha=i,tol=1)
    
    score=cross_val_score(clf,xtr,ytr,cv=5)
   
    b.append(np.mean(score))
#Ridge  
for i in a:
    clf = linear_model.Ridge(alpha=i)
    score=cross_val_score(clf,xtr,ytr,cv=5)
   
    b.append(np.mean(score))
    
# Fitting best model for test error   
n = linear_model.Ridge(alpha=10)
n.fit(xtr,ytr)
y=n.predict(xte)
print ("test Error for Linear MOdel",compute_error(y,yte)) # Test Error
#Kaggle prediction
n.fit(trainx,trainy)
rnnt=n.predict(testx)
rnnkout=kaggle.kaggleize(rnnt,'kagglernnout')

############################################################################
#SVM


from sklearn.preprocessing import StandardScaler #Normalising the data
X_train = StandardScaler().fit_transform(xtr)
X_test = StandardScaler().fit_transform(xte)
from sklearn.svm import SVR
clf = SVR(degree=1,kernel='poly') #Degree 1 with poly
y=clf.predict(X_test)
print ("test error for poly kernel with 1 degree",compute_error(y,yte))

clf = SVR(degree=1,kernel='poly')
score=cross_val_score(clf,X_train,ytr)
print ("out of sample error for poly kernel with 1 degree",np.mean(score))


clf = SVR(degree=2,kernel='poly')
score=cross_val_score(clf,X_train,ytr)
print ("out of sample error for poly kernel with 2 degree",np.mean(score))


clf = SVR(degree=1,kernel='rbf')
score=cross_val_score(clf,X_train,ytr)
print ("out of sample error for rbf kernel with 1 degree",np.mean(score))


clf = SVR(degree=2,kernel='rbf')
score=cross_val_score(clf,X_train,ytr)
print ("out of sample error for rbf kernel with 2 degree",np.mean(score))

clf = SVR(degree=2,kernel='rbf')
clf.fit(X_train,ytr)
y=clf.predict(X_test)
print("Test error for best model",compute_error(y,yte))#test error

#Kaggle Submission
X_test = StandardScaler().fit_transform(testx)# Normalising the data
X_train=StandardScaler().fit_transform(trainx)
#Y_train=StandardScaler().fit_transform(trainy)
clf.fit(X_train,trainy)
ypsvm=clf.predict(X_test)
svmkout=kaggle.kaggleize(ypsvm,'svmnnout')
#############################################################################
#Neural Network
from sklearn.neural_network import MLPRegressor
a=[10,20,30,40] #Cv with different values
b=[]
for i in a:
    mlp=MLPRegressor(hidden_layer_sizes=i,activation='logistic',max_iter=600)
    score=cross_val_score(mlp,X_train,trainy)
    b.append(np.mean(score))



# test error calculation
X_train = StandardScaler().fit_transform(xtr)
X_test = StandardScaler().fit_transform(xte)
bb=[]
for i in a:
    mlp=MLPRegressor(hidden_layer_sizes=i,activation='relu',max_iter=600)
    mlp.fit(X_train,ytr)
    yy=mlp.predict(X_test)
    bb.append(compute_error(yy,yte))
  
#kaggle prediction
X_test = StandardScaler().fit_transform(testx)
X_train=StandardScaler().fit_transform(trainx)

clf = SVR(degree=2,kernel='rbf')
a=[3,5,7]  
b=[]
for i in a:  
    score=cross_val_score(clf,X_train,trainy,cv=i)
    b.append(np.mean(score))
clf = SVR(degree=2,kernel='rbf')
clf.fit(X_train,trainy)
out=clf.predict(X_test)
kaggle.kaggleize(out,'best.csv')    
# cv=3 gives the best value


    
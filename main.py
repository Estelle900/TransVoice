import numpy as np
import random
import os
import math
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import cv2
import csv
from scipy.fftpack import dct
from scipy.io import wavfile as wav
from yellowbrick.classifier import ConfusionMatrix
import librosa
from playsound import playsound

#Sample Data
filename="archive\\voice.csv"
rows=[]
# reading csv file
with open(filename, 'r') as csvfile:    
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    
    # extracting field names through first row
    fields = next(csvreader)
 
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)
why=[]
X=[]
for i in range(len(rows)):
    X.append([float(x) for x in rows[i][1:20]])
    if rows[i][-1]=="male":
        why.append(1)
    elif rows[i][-1]=="female":
        why.append(0)


#Standardise data
Xstd=(X-np.mean(X,axis=0))/np.std(X,axis=0)
#Split data
X_train, X_test, y_train, y_test = train_test_split(Xstd, why, random_state=1, test_size=0.2)
#Analysis

def pca(X, n_components):
    """X: Standardized dataset, observations on rows
     n_components: dimensionality of the reduced space
     labels: targets, for visualization"""

    # calculate eigenvalues
    X_cov = np.cov(X.T)
    e_values, e_vectors = np.linalg.eigh(X_cov)

    # Sort eigenvalues and their eigenvectors in descending order
    e_ind_order = np.flip(e_values.argsort())
    e_values = e_values[e_ind_order]
    e_vectors = e_vectors[:, e_ind_order] # note that we have to re-order the columns, not rows
    
    # now we can project the dataset on to the eigen vectors (principal axes)
    prin_comp_evd = X @ e_vectors

    # sklearn
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(X)
    return reduced,pca,e_vectors,e_values
        
#Apply PCA
#reduced_dim,pca_matrix,e_vectors,e_values=pca(X_train,19)

#NN
'''
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
# evaluate model
scores = cross_val_score(model, reduced_dim, ds.target, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
x.append(np.mean(scores))
'''


#Train model
X_train=np.transpose(X_train)
X_test=np.transpose(X_test)
X_train2=[]
X_test2=[]
#clf = MLPClassifier(hidden_layer_sizes=(640,640,640),activation="tanh" ,random_state=1, max_iter=200000).fit(reduced_dim, y_train)
'''
for i in range(len(X_train)):
    X_train2.append(X_train[i][:6])
for i in range(len(X_test)):
    X_test2.append(X_test[i][:6])
'''
acc=[0.8107255520504731, 0.6167192429022083, 0.88801261829653, 0.5457413249211357, 0.8911671924290221, 0.6719242902208202, 0.6324921135646687, 0.7665615141955836, 0.7066246056782335, 0.7413249211356467, 0.6482649842271293, 0.9574132492113565, 0.5488958990536278, 0.5362776025236593, 0.6198738170347003, 0.637223974763407, 0.6293375394321766, 0.6309148264984227, 0.5189274447949527]
'''
for i in range(len(X_train)):
    X_train2=X_train[i]
    X_test2=X_test[i]
    clf = MLPClassifier(hidden_layer_sizes=(640,640,640),activation="relu" ,random_state=1, max_iter=200000).fit(np.transpose([X_train2]), y_train)
    c=clf.score(np.transpose([X_test2]), y_test)
    acc.append(c)
    print(c)
print(acc)
'''
fig,ax=plt.subplots()
ax.bar(fields[:19],acc)
ax.set_title("Results per feature")
#ax.xlabel("Feature")
#ax.ylabel("Accuracy")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.show()

#Test
'''
#y_pred=clf.predict(pca_matrix.fit_transform(X_test))
#print("The training accuracy is ", (clf.score(pca_matrix.fit_transform(X_train), y_train)))
#print("The test accuracy is ", (clf.score(pca_matrix.fit_transform(X_test), y_test)))
print("The test accuracy with no dim reduction is ", clf.score(X_test2, y_test))
'''
'''
for i in X_test2:
    print(i,clf.predict([i]))
'''
#Confusion matrix
'''
cm=ConfusionMatrix(clf, classes=[0,1])
#cm.score(pca_matrix.fit_transform(X_test), y_test)
cm.score(X_test, y_test)
cm.show()
'''


#Graphing dim-reduced models
'''
graphx=[]
graphy=[]
for i in range(1,20):
    reduced_dim,pca_matrix,trash1,trash2=pca(X_train,i)
    clf = MLPClassifier(hidden_layer_sizes=(640,640,640),activation="relu" ,random_state=1, max_iter=200000).fit(reduced_dim, y_train)
    graphy.append(clf.score(pca_matrix.fit_transform(X_test), y_test))
    graphx.append(i)
    
fig,ax=plt.subplots()
ax.scatter(graphx,graphy)
ax.set_title("n-dimensional PCA model results")
ax.xlabel("Dimensions")
ax.ylabel("Accuracy")
plt.show()
'''

#eigenvector heatmap
'''
for i in range(2):
    e_vectors[i]=e_vectors[i]*pca_matrix.explained_variance_ratio_[i]

fig,ax=plt.subplots()
im=ax.imshow(e_vectors, cmap = 'autumn' , interpolation = 'nearest' )
fig.tight_layout()
#axes
axe=np.arange(20)
ax.set_xticks(axe,fields[:20])
ax.set_yticks(axe,axe)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
ax.set_title("Contributions of components to eigenvectors")
plt.show()
'''

#factor analyses

'''
factors=[]
for i in range(19):
    a=0
    for j in range(2):
        a+=e_vectors[j][i]**2
    factors.append(math.sqrt(a))
fig,ax=plt.subplots()
ax.bar(np.arange(19),np.abs(factors),tick_label=fields[:19])
ax.set_title("Approximate significance of components in gender perception")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
plt.show()
'''

#Extract audio features
for f in os.listdir("clips"):
    if f[-4:]==".wav":
        #FFT
        features=[]
        rate,data=wav.read("clips\\"+f)
        FFT=np.abs(dct(data)) 
        fig,ax=plt.subplots()
        data=np.arange(1,rate/2,rate/len(FFT))
        FFT=FFT[:len(data)]
        print("success")
        #plot fft
        #ax.vlines(data,0,FFT[:len(data)])
        #plt.show()
        i=0
        data=data/1000
        crit=0
        while crit==0:
            if data[i]>0.28:
                crit=i
            i+=1
        data=data[:crit]
        FFT=FFT[:crit]

        #mean freq
        features.append((FFT@data/sum(FFT)))

        #sd freq
        features.append(math.sqrt(FFT@np.square(data-features[0])/sum(FFT)))

        #median freq
        data.sort()
        features.append(data[round(len(data)/2)])

        #q25
        features.append(data[round(len(data)/4)])

        #q75
        features.append(data[round(len(data)*3/4)])

        #interquartile r
        features.append(features[-1]-features[-2])
        print(features)
        #skewness
        #features.append((data@np.power(data-features[0],3))/features[1]**3/sum(FFT))
        #return features
        '''
        #kurtosis
        n=np.square(np.square(data-features[0]))
        n=FFT@n/sum(FFT)
        d=np.square(FFT@np.square(data-features[0])/sum(FFT))
        features.append(n/d) #may have an error, kurt>3 in dataset (and is theoretical value for gaussian dist)

        #spectral entropy
        P=np.square(FFT)/len(FFT)
        P=P/sum(P)
        pse=0
        for i in P:
            pse-=i*math.log(i,2)
        print(pse) #error, shld be less than 1, /10 is a stopgap

        #spectral flatness

        '''
        for i in range(len(X)):
            X[i]=X[i][:6]
        features=(features-np.mean(X,axis=0))/np.std(X,axis=0)
        #test
        playsound("clips\\"+f)
        print(features)
        print(clf.predict([features]))

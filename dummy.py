import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import keras

x=[]
y=[]

X=np.load('X.npy')
Y=np.load('Y.npy')
for i in range(len(X)):
    x.append(X[i])
    y.append(Y[i])

##model = keras.models.load_model('wgan_model_normal.h5')
##noise = np.random.normal(0, 1, (1000, 100))
##gen_norm = model.predict(noise)
##for i in gen_norm:
##    x.append(i)
##    y.append([1,0])
##
##model = keras.models.load_model('wgan_model_autistic.h5')
##noise = np.random.normal(0, 1, (1000, 100))
##gen_norm = model.predict(noise)
##for i in gen_norm:
##    x.append(i)
##    y.append([0,1])

x=np.array(x)
y=np.array(y)
y=y[:,0]

X_columns = ['mean', 'standard deviation', 'kurt', 'skewness']
Y_columns = ['label']
X = pd.DataFrame(columns = X_columns)
Y = pd.DataFrame(columns = Y_columns)

v=[]
for i in range(len(x)):
  a = np.array([np.mean(x[i]),
                       np.std(x[i]),
                       np.mean(kurtosis(x[i])),
                       np.mean(skew(x[i]))])
  v.append(a)
  X.loc[i]=a
  Y.loc[i] = y[i]





X=np.array(v)
Y=y
from sklearn.utils import shuffle
X,Y=shuffle(X,Y)
limit=int(len(X)*0.8)
input_dim = X.shape[1]


from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU , Dropout , Activation, Embedding, LSTM, SimpleRNN, Flatten, Reshape, Conv1D,BatchNormalization, MaxPooling1D

model = Sequential()
model.add(Dense(800, activation='relu', input_dim=input_dim))
model.add(Dropout(0.8))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(800, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='tanh'))
model.summary()
model.compile(loss='mse', optimizer='adam',
              metrics=['mae', 'acc'])

history=model.fit(X[:limit], Y[:limit], epochs=20
    , batch_size=1000, validation_data=(X[limit:], Y[limit:]), verbose=1)



X_train=X[:limit]
y_train=Y[:limit]
X_test=X[limit:]
y_test=Y[limit:]

print('******** SVM ************')
from sklearn import svm
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
print('training acc=')
print(str(clf.score(X_train, y_train))+'\n')
print('validation acc=')
print(str(clf.score(X_test, y_test))+'\n')

##print('********* K means ***********')
##from sklearn.cluster import KMeans
##kmeans = KMeans(n_clusters=2)
##kmeans.fit(X_train)
##
##clf.predict(eeg_emb[-1])
##kmeans.predict(eeg_emb[-1])



print('********* KNN ***********')
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
print('training acc=')
print(str(neigh.score(X_train, y_train))+'\n')
print('validation acc=')
print(str(neigh.score(X_test, y_test))+'\n')
print('********************')


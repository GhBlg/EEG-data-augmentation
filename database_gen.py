import numpy as np
import os
import mne

def label(i):
    word_label=i.split('.')[-2][-1]   
    if word_label == 'n': return [1,0]
    elif word_label == 'a': return [0,1]

def database():
    Y=[]
    X=[]

    #19 sec
##################    #BONN dataset
##    folder='C:\\Users\\ghait\\Desktop\\data\\bonndata'
##    f1=0
##    for f in os.listdir(folder):
##        f1=f1+1
##        print(f)
##        a=open(folder+'\\'+f,'r')
##        a1=a.readlines()
##        a.close()
##        V=[]    
##        for i in a1[0][1:-1].split(','):
##            if i[0]=='[' :
##                i=i[1:]
##            if i[0]==' ' :
##                if i[1]=='[':
##                    i=i[2:]
##            if i[-1]==']':
##                i=i[:-1]
##            
##            V.append(int(i))
##            if len(V)==17361:
##                Y.append(f1)
##                B.append(V)
##                C.append(V)
##                D.append(V)
##                V=[]

######################    #KSU dataset
    folder='C:\\Users\\ghait\\Desktop\\ksu edf\\'
    for f in os.listdir(folder):
        raw = mne.io.read_raw_edf(folder+f)
        sfreq=256
        steps=2560
        
        i=0
        while i<len(raw)-steps:
            data, times = raw[:, i:i+steps]
            i=i+sfreq*10
            X.append(data)
            Y.append(label(f))

#######################    #MIT dataset
##    folder='C:\\Users\\ghait\\Desktop\\mit edf\\'
##    for f in os.listdir(folder):
##        raw = mne.io.read_raw_edf(folder+f)
##        sfreq=256
##        steps=256#4800
##        
##        i=0
##        while i<len(raw)-steps:
##            data, times = raw[1:17, i:i+steps]
##            i=i+sfreq*100
##            X.append(data)
##            Y.append(f)

    X=np.array(X)
    
    return X,Y


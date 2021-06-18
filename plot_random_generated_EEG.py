import keras
import numpy as np
from spectrumtest import pltspect
from topomap import topoplot
import matplotlib.pyplot as plt
import pandas as pd
import mne



raw = mne.io.read_raw_edf('1n.edf')
raw.ch_names[0]='FP1'
raw.ch_names[3]='T7'
raw.ch_names[4]='P7'


model = keras.models.load_model('wgan_model_normal.h5')
##for i in range(15):
noise = np.random.normal(0, 1, (1000, 100))
gen_imgs = model.predict(noise)
pltspect(gen_imgs[2][5],256,10)
##
##    d=dict(zip(raw.ch_names,np.mean(gen_imgs[i],axis=1)*0.01))
##    topoplot(d)
##    plt.show()

    
##for i in range(20,30):
##    d=dict(zip(raw.ch_names,np.mean(X[i],axis=1)))
##    topoplot(d)
##    plt.show()

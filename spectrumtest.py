from scipy import fft
from matplotlib.pylab import *
import dtcwt
import os
import  numpy as np

def pltspect(y,Fs,time):
    Ts = 1.0/Fs; # sampling interval
    #time=120
    aa=time*Fs
    t = np.arange(0,time,Ts) # time vector


    def plotSpectrum(y,Fs):
     global freqz, amps , Z  ,frrq
     """
     Plots a Single-Sided Amplitude Spectrum of y(t)
     """
     n = len(y) # length of the signal
     k = arange(n)
     T = n/Fs
     frq = k/T # two sides frequency range
     frrq=frq
     frq = frq[range(int(n/2))] # one side frequency range

     Y = fft(y)/n # fft computing and normalization
     Z=Y*n
     Y = Y[range(int(n/2))]
     
     plot(frq,abs(Y),'r') # plotting the spectrum
     xlabel('Freq (Hz)')
     ylabel('|Y(freq)|')
     amps=abs(Y)
     freqz=frq
     



    subplot(2,1,1)
    plot(t,y)
    xlabel('Time (S)')
    ylabel('Amplitude')
    subplot(2,1,2)
    plotSpectrum(y,Fs)
    show()
    return freqz, amps, Z ,frrq

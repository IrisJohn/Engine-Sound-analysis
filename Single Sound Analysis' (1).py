#!/usr/bin/env python
# coding: utf-8

# In[4]:



import librosa
song, sr1 = librosa.load(r"C:\Users\irisj\UST Global Tech\Audi.wav")
print(len(song)," ",sr1)


# In[5]:


song


# In[6]:


#for plotting the signal
from matplotlib import pyplot as plt
import librosa.display
plt.figure(figsize=(30, 4))
librosa.display.waveplot(song, sr=sr1)


# In[7]:


#applying stft windowing technique
import scipy
import scipy.signal
f, t,Zxx= scipy.signal.stft(song, fs=sr1, window='hamming', nperseg=180,
                        noverlap=None, nfft=2048, detrend=False, return_onesided=True, padded=True,
                        axis=-1)
f,t,Zxx

#f contains the array of frquencies
#t contains time domain segments
#Zxx contains stft value


# In[8]:


#plotting spectrigram
X = librosa.stft(song)
Xdb = librosa.amplitude_to_db(X)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()
plt.savefig("figure.png")


# In[9]:


import numpy as np 
data,fs = librosa.load(r'C:\Users\irisj\UST Global Tech\Bummo.wav')


# In[10]:


print(len(data)," ",fs)


# In[11]:


import numpy as np 
data,fs = librosa.load(r'C:\Users\irisj\UST Global Tech\Bummo.wav')

zero=np.zeros(847497)
zero1=np.zeros(847498)
# zero=np.zeros(28769)
# zero1=np.zeros(28770)
new=np.append(data,zero)
#data=new
data=np.append(zero1,new)
plt.figure(figsize=(30, 4))
librosa.display.waveplot(data, sr=fs)
print(len(data)," ",fs)


# In[12]:


import scipy
import scipy.signal
f_song1, t_song1,Zxx_song1= scipy.signal.stft(data, fs=sr1, window='hamming', nperseg=180,
                        noverlap=None, nfft=2048, detrend=False, return_onesided=True, padded=True,
                        axis=-1)
print(len(f_song1)," ",len(t_song1)," ",len(Zxx_song1))


# In[20]:



X_noise_song = librosa.stft(data)
Xdb = librosa.amplitude_to_db(X_noise_song)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()
plt.savefig('mixed noise spectro.png')


# In[14]:


#to mix the signals ad them
#song #1025   30066   1025


noise_song = song + data

noise_song


# In[15]:


import scipy
import scipy.signal
f_song, t_song,Zxx_song= scipy.signal.stft(noise_song, fs=sr1, window='hamming', nperseg=180,
                        noverlap=None, nfft=2048, detrend=False, return_onesided=True, padded=True,
                        axis=-1)
print(len(f_song)," ",len(t_song)," ",len(Zxx_song))


# In[16]:


#mixed signal spectrogram

X_noise_song = librosa.stft(noise_song)
Xdb = librosa.amplitude_to_db(X_noise_song)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()


# In[17]:


import IPython.display as ipd
ipd.Audio(noise_song,rate=sr1)


# In[18]:



import IPython.display as ipd
ipd.Audio("Audi.wav")


# In[22]:


import librosa
signal, sr = librosa.load("download.wav")
print(len(signal)," ",sr)


# In[23]:


#for mixed signal
from matplotlib import pyplot as plt
import librosa.display
plt.figure(figsize=(30, 4))
librosa.display.waveplot(signal, sr=sr)


# In[24]:



import scipy
import scipy.signal
f, t,Zxx= scipy.signal.stft(signal, fs=sr, window='hamming', nperseg=180,
                        noverlap=None, nfft=2048, detrend=False, return_onesided=True, padded=True,
                        axis=-1)
print(len(f)," ",len(t)," ",len(Zxx))


# In[25]:


#basically stft represents a signal in time-frequency domain 
#so after doing stft it return a complex matrix

#


# In[26]:


X = librosa.stft(signal)
Xdb = librosa.amplitude_to_db(X)#converts into db-sclaed spectrogram
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.clim(-60,10)
plt.colorbar()


# In[27]:


X#the real spectrogram array


# In[28]:


#seperate the spectrogram  into magnitude and phase components
X, X_phase = librosa.magphase(X_noise_song)
n_components = 2
W, H = librosa.decompose.decompose(X, n_components=n_components, sort=True)


# In[29]:


#librosa.decopose
#decomposes the spectrogram matrix into components


# In[30]:


print(W.shape)
print(H.shape)


# In[31]:


#for W matrix
import numpy
plt.figure(figsize=(30,4))
logW = numpy.log10(W)
for n in range(n_components):
    plt.subplot(numpy.ceil(n_components/2.0), 2, n+1)
    plt.plot(logW[:,n])
    plt.ylim(-2, logW.max())
    plt.xlim(0, W.shape[0])
    plt.ylabel('Component %d' % n)
    plt.tight_layout()


# In[32]:


#for Hmatrix
plt.figure(figsize=(30,4))
for n in range(n_components):
    plt.subplot(numpy.ceil(n_components/2.0), 2, n+1)
    plt.plot(H[n])
    plt.ylim(0, H.max())
    plt.xlim(0, H.shape[1])
    plt.ylabel('Component %d' % n)


# In[33]:


#1.spectrogram of the orifinal engine sound
X = librosa.stft(song)
Xdb = librosa.amplitude_to_db(X)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()

#2.Spectrogram for the music noise song
X_noise_song = librosa.stft(data)
Xdb = librosa.amplitude_to_db(X_noise_song)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()

#3.Spectrogram for the mixed noise
X_noise_song = librosa.stft(noise_song)
Xdb = librosa.amplitude_to_db(X_noise_song)
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
plt.clim(-60, 10)
plt.colorbar()

#computes the outer product first
#converts into db scale
#showing the final spectrogram for components

for n in range(n_components):
    Y = scipy.outer(W[:,n], H[n])*numpy.exp(1j*numpy.angle(X_noise_song))
    Xdb = librosa.amplitude_to_db(Y)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr1, x_axis='time', y_axis='hz')
    plt.clim(-60,10)
    plt.colorbar()


# In[34]:



import IPython
reconstructed_signal = scipy.zeros(847498)
components = list()
for n in range(n_components):
    Y = scipy.outer(W[:,n], H[n])*numpy.exp(1j*numpy.angle(X_noise_song))
    y = librosa.istft(Y)
    components.append(y)
    reconstructed_signal[:len(y)] += y
    IPython.display.display( IPython.display.Audio(y, rate=fs) )


# In[ ]:





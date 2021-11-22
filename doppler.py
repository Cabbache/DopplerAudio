import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

fs, sound = wavfile.read('ambulance.wav')
ch1 = sound[:,0]
ch1 = ch1 / 2**15
ch1 = ch1[100000:700000]

#fs = 44100
#ch1 = (np.sin(2*np.pi*np.arange(80000)*2000/fs)).astype(np.float32)

print(fs)

num_samples = ch1.size
V = 330
A = -6
B = 6
C = 1
X = np.arange(A,B,(B-A)/(num_samples))
vels = 60*X/((C**2 + X**2)**0.5)
f = (V - vels) / V

#plt.plot(f)
#plt.show()

doppler = np.zeros(num_samples)
delta = 1.0 / fs
index = 0
indices = (1/f)
for i in range(num_samples):
	value = ch1[i]
	if index*fs >= num_samples-1:
		break
	pos = index*fs
	mod = pos % 1.0
	doppler[int(pos)] += value * (1-mod)
	doppler[int(pos)+1] += value * mod
	index += delta * indices[i]
N = doppler.size

#apply amplitude fade in and fade out (it's linear, TODO: inverse square)
doppler = doppler*np.concatenate((np.arange(N/2), np.arange(N/2,0,-1)))/(N/2)

#normalize signal scale to -1, 1
doppler = -1 + (2*(doppler-np.min(doppler)) / (np.max(doppler)-np.min(doppler)))

wavfile.write("modded.wav", fs, doppler)
wavfile.write("unmodded.wav", fs, ch1)

import time
from wsgiref.headers import tspecials
import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal
import scipy

sample_rate = 40e6 #Hz
fc = 2480e6 #Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.gain_control_mode = 'manual' 
sdr.sample_rate = int(sample_rate) 
sdr.rx_rf_bandwidth = int(sample_rate) 
sdr.rx_lo = int(fc) 
sdr.rx_hardwaregain_chan0 = 70.0
sdr.rx_buffer_size = 65536

counter = 0

data_array = sdr.rx()

f, t, Sxx = signal.spectrogram(data_array, sample_rate, return_onesided=False)
f = np.fft.fftshift(f)+fc
Sxx = np.fft.fftshift(Sxx, axes=0,)
Sxx = np.transpose(Sxx)
Sxx = np.flipud(Sxx)

out_data = [np.abs(x)*np.sign(np.angle(x)) for x in data_array]
time_domain = np.linspace(0,sdr.rx_buffer_size/sample_rate,len(data_array))
freq_domain = np.linspace(fc-sample_rate/2,fc+sample_rate/2, sdr.rx_buffer_size)

data_array_fft_shifted= np.abs(np.fft.fftshift(np.fft.fft((data_array))))


#Coarse Frequency Correction

#Plot time domain and spectrogram

#plt.subplot(2,1,1)
#plt.plot(time_domain, out_data)
#plt.xlabel("Time [sec]")
#plt.ylabel("Magnitude")

#plt.subplot(2,1,2)
#plt.pcolormesh(f, t, Sxx, shading="gouraud")
#plt.xlabel("Freqeuency [Hz]")
#plt.ylabel("Time [sec]")
#plt.show()

# This is for testing only
#print(offset)
#samples_of_f_1 = np.abs(np.fft.fftshift(np.fft.fft((samples_shifted))))
#fig, (plotT, plotF) = plt.subplots(2)
#plotT.plot(freq_domain,  samples_of_f_1)
#plotF.plot(freq_domain, data_array_fft_shifted)
#plt.show()

#Fine Frequency Correction

#DPLL
#PED

#Ideal FSK from Wyglinski
# Define radio parameters
Rsymb = 1e6  # BLE symbol rate
Rsamp = sample_rate # Sampling rate
N = len(out_data)  # Total number of signal samples in demo
Foffset = 1.0e6  # Expected frequency offset of FSK tones from signal carrier frequency (Hz)
PhaseOffset = 0.0  # Initial phase offset of FSK modulation (radians)

# Generate time indices
t = np.linspace(0.0,(N-1)/(float(Rsamp)),N)  

# Generate ideal I/Q signal constellation points without unexpected frequency offset
deltaF = 0.0 # Unexpected frequency offset set to zero
dataI = np.cos(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Inphase data samples
dataQ = -np.sin(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Quadrature data samples

ideal = dataI + 1j*dataQ
#euclidean_distance = scipy.integrate((ideal - out_data) ** 2)
phase_ideal = np.angle(ideal)
phase_real = np.angle(out_data)
for i in range(len(out_data)):
    phase_error = phase_ideal[i] - phase_real[i]
    print(phase_error)

#LPF
#NCO

#Demo from Wyglinski
# fsk_dpll.py
# 
# Demo of simple DPLL module operating on simulated FSK data
#
# A. M. Wyglinski (alexw@wpi.edu), 2021.03.06
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
# Define radio parameters
Rsymb = 1e6  # BLE symbol rate
Rsamp = 20.0e6 # Sampling rate
N = int(1e3)  # Total number of signal samples in demo
Foffset = 1.0e6  # Expected frequency offset of FSK tones from signal carrier frequency (Hz)
PhaseOffset = 0.0  # Initial phase offset of FSK modulation (radians)
# Generate time indices
t = np.linspace(0.0,(N-1)/(float(Rsamp)),N)  
# Generate ideal I/Q signal constellation points without unexpected frequency offset
deltaF = 0.0 # Unexpected frequency offset set to zero
dataI = np.cos(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Inphase data samples
dataQ = -np.sin(2.0*np.pi*(Foffset+deltaF)*t+PhaseOffset*np.ones(N)) # Quadrature data samples
# Plot signal constellation diagram
plt.figure(figsize=(9, 5))
plt.plot(dataI,dataQ)
plt.xlabel('Inphase')
plt.ylabel('Quadrature')
plt.show()

import time
import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal

sample_rate = 40e6 #Hz
fc = 2450e6 #Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.gain_control_mode = 'manual' 
sdr.sample_rate = int(sample_rate) 
sdr.rx_rf_bandwidth = int(sample_rate) 
sdr.rx_lo = int(fc) 
sdr.rx_hardwaregain_chan0 = 70.0
sdr.rx_buffer_size = 8192

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

samples_of_f= np.abs(np.fft.fftshift(np.fft.fft((data_array))))

plt.subplot(2,1,1)
plt.plot(time_domain, out_data)
plt.xlabel("Time [sec]")
plt.ylabel("Magnitude")

plt.subplot(2,1,2)
plt.pcolormesh(f, t, Sxx, shading="gouraud")
plt.xlabel("Freqeuency [Hz]")
plt.ylabel("Time [sec]")
plt.show()

#Coarse Frequency Correction
midpoint_sum = 0.5*np.sum(samples_of_f)
indexing_sum = 0
sum_extrema = 0
f_c = int(0.5*len(freq_domain))

while (midpoint_sum <= indexing_sum):
    indexing_sum = np.sum(samples_of_f, initial=0, where=sum_extrema)
    sum_extrema += 1
offset = freq_domain[sum_extrema] - freq_domain[f_c]
samples_shifted = data_array * np.exp(-1j*2*np.pi*offset*time_domain)

# This is for testing only
# print(offset)
# samples_of_f_1 = np.abs(np.fft.fftshift(np.fft.fft((samples_shifted))))
# fig, (plotT, plotF) = plt.subplots(2)
# plotT.plot(freq_domain, freqSamplesPrime)
# plotF.plot(freq_domain, freqSamples)
# plt.show()

#Fine Frequency Correction

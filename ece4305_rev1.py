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
# data = np.abs(data_array)
# phase = np.angle(data_array)

# #doesn't work becasue you're changing the whole data array
# for i in phase:
#    if i < 0:
#       data = np.abs(data_array)*(-1)
#    else:
#       data = np.abs(data_array)*(+1)

# for i in range(len(phase)):
#     if phase[i] < 0:
#         data[i] = data[i] * -1
#     else:
#         data[i] = data[i] * 1

# for i in range(len(phase)):
#     if phase[i] < 0:
#         data[i] = data[i] * -1
    
# out_data = []
# for i in data_array:
#     out_data.append(np.abs(i)*np.sign(np.phase(i)))

out_data = [np.abs(x)*np.sign(np.angle(x)) for x in data_array]
time_domain = np.linspace(0,sdr.rx_buffer_size/sample_rate,len(data_array))

plt.subplot(2,1,1)
plt.plot(time_domain, out_data)
plt.xlabel("Time [sec]")
plt.ylabel("Magnitude")

plt.subplot(2,1,2)
plt.pcolormesh(f, t, Sxx, shading="gouraud")
plt.xlabel("Freqeuency [Hz]")
plt.ylabel("Time [sec]")
plt.show()

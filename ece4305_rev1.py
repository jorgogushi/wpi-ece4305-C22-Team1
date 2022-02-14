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

plt.subplot(2,1,1)
plt.plot(time_domain, out_data)
plt.xlabel("Time [sec]")
plt.ylabel("Magnitude")

plt.subplot(2,1,2)
plt.pcolormesh(f, t, Sxx, shading="gouraud")
plt.xlabel("Freqeuency [Hz]")
plt.ylabel("Time [sec]")
plt.show()

mu = 0 # initial estimate of phase of sample
out = np.zeros(len(samples) + 10, dtype=np.complex)
out_rail = np.zeros(len(samples) + 10, dtype=np.complex) # stores values, each iteration we need the previous 2 values plus current value
i_in = 0 # input samples index
i_out = 2 # output index (let first two outputs be 0)
while i_out < len(samples) and i_in < len(samples):
    out[i_out] = samples[i_in + int(mu)] # grab what we think is the "best" sample
    out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
    x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
    y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
    mm_val = np.real(y - x)
    mu += sps + 0.3*mm_val
    i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
    mu = mu - np.floor(mu) # remove the integer part of mu
    i_out += 1 # increment output index
out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
samples = out 

samples = samples**2
psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
f = np.linspace(-fs/2.0, fs/2.0, len(psd))
plt.plot(f, psd)
plt.show()

max_freq = f[np.argmax(psd)]
Ts = 1/fs # calc sample period
t = np.arange(0, Ts*len(samples), Ts) # create time vector
samples = samples * np.exp(-1j*2*np.pi*max_freq*t/2.0)

import time
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
import sys as sys
import scipy.integrate as integrate
from scipy.signal import butter,lfilter, freqz, filtfilt

data_array = np.loadtxt("/Users/jorgogushi/Desktop/data_array_20fs_2426fc_1", dtype=np.cdouble)


sample_rate = 20e6 #Hz
fc = 2426e6 #Hz
buffer_size = 2**12
start = int(8e5)
end = int(8e5+buffer_size)
print(start)
print(end)
data_array = data_array[start:end]

f, t, Sxx = signal.spectrogram(data_array, sample_rate, return_onesided=False)
f = np.fft.fftshift(f)+fc
Sxx = np.fft.fftshift(Sxx, axes=0,)
Sxx = np.transpose(Sxx)
Sxx = np.flipud(Sxx)

out_data = [np.abs(x)*np.sign(np.angle(x)) for x in data_array]
time_domain = np.linspace(0,buffer_size/sample_rate,len(data_array))
freq_domain = np.linspace(fc-sample_rate/2,fc+sample_rate/2,buffer_size)

shifted_fft= np.abs(np.fft.fftshift(np.fft.fft((data_array))))

plt.subplot(2,1,1)
plt.plot(time_domain, out_data)
plt.xlabel("Time [sec]")
plt.ylabel("Magnitude")

plt.subplot(2,1,2)
plt.pcolormesh(f, t, Sxx, shading="gouraud")
plt.xlabel("Freqeuency [Hz]")
plt.ylabel("Time [sec]")
plt.show()

#Coarse Frequency Correction Trial 1
#integration = abs(integrate.simpson(shifted_fft))/2
#for i in range(len(shifted_fft)):
#   index_integration = integrate.simpson(shifted_fft[0:i+1])
#
#   if index_integration > integration:
#       freq_offset = abs(fc - )

#Coarse Frequency Correction Trial 2
f_c = int(0.5*len(freq_domain))

half_int_stop_point = 0.5*np.sum(shifted_fft)
current_rolling_integral = 0
for i in range(len(shifted_fft)):
    current_rolling_integral += shifted_fft[i]
    if current_rolling_integral > half_int_stop_point:
        #calculate f_offset based on i here
        freq_offset = freq_domain[i] - fc
        break

samples_shifted = data_array*np.exp(1j*2*np.pi*freq_offset*time_domain)

#print(freq_offset)
#samples_of_f_1 = np.abs(np.fft.fftshift(np.fft.fft((samples_shifted))))
#fig, (plotT, plotF) = plt.subplots(2)
#plotT.plot(freq_domain,  samples_of_f_1)
#plotF.plot(freq_domain, shifted_fft)
#plt.show()

#Isolating One Packet
#samples_peak = np.argmax(data_array) # Finds the peak samples
#samples_packet = data_array[samples_peak - round((350e-6*sample_rate)): samples_peak+round((350e-6*sample_rate))]
#t_s_packet = np.linspace(0, (len(samples_packet)/sample_rate),len(samples_packet))

# DIGITAL PHASE LOCK LOOP #

# Phase Error Detector (PED)

samples_of_dpll = np.zeros(len(samples_shifted))
phase = np.zeros(len(samples_shifted))

for i in range(len(samples_shifted)):
    print(i)
    for j in range(len(samples_shifted)):
        phase[j] = np.angle(samples_shifted[j])
        if -np.pi<phase[j]<-np.pi/2:
            phase[j] = -np.pi - phase[j]
        if np.pi/2<phase[j]<np.pi:
            phase[j] = np.pi - phase[j]
    avg = phase[i-1]*0.4 + phase[i-2]*0.3 + phase[i-3]*0.2 + phase[i-4]*0.1
    samples_of_dpll[i] = samples_shifted[i] * np.exp(-1j*2*np.pi*avg)

#Plotting
plt.scatter(np.real(samples_shifted), np.imag(samples_shifted), marker = '.')
plt.scatter(np.real(samples_of_dpll), np.imag(samples_of_dpll), marker = 'x', color = 'r')
plt.title('IQ Samples')
plt.show()

plt.scatter(time_domain, np.angle(samples_of_dpll),marker = 'x')
plt.title('Time Domain vs. Phase')
plt.show()

# Change in Phase
phase_array = np.zeros(len(samples_of_dpll))

for k in range(len(samples_of_dpll)):
    phase_array[k] = np.angle(samples_of_dpll[k])
    if phase_array[k]<0:
        phase_array[k]=phase_array[k]+2*np.pi 
    phase_array[k] = phase_array[k] - phase_array[k-1]

plt.plot(time_domain,phase_array)
plt.show()

#p_change = np.zeros(len(samples_of_dpll))
#for m in range(p_change[1:len(p_change)]):
#    p_change = phase_array[k]-phase_array[k-1]

#plt.scatter(time_domain,p_change,marker = 'x')
#plt.show()

# Lowpass Filter (LPF)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(phase_array, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, phase_array)
    return y


# Setting standard filter requirements.
order = 1
fs = sample_rate       
cutoff = 3.667  

b, a = butter_lowpass(cutoff, fs, order)

# Plotting the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Filtering and plotting
y = butter_lowpass_filter(phase_array, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(time_domain, phase_array, 'b-', label='data')
plt.plot(time_domain, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()



import time
from wsgiref.headers import tspecials
import numpy as np
import adi
import matplotlib.pyplot as plt
from scipy import signal
import scipy

#Set sample rate and center frequency
sample_rate = 40e6 #Hz
fc = 2480e6 #Hz

#Set up data collection from the PLUTO
sdr = adi.Pluto("ip:192.168.2.1")
sdr.gain_control_mode = 'manual' 
sdr.sample_rate = int(sample_rate) 
sdr.rx_rf_bandwidth = int(sample_rate) 
sdr.rx_lo = int(fc) 
sdr.rx_hardwaregain_chan0 = 70.0
sdr.rx_buffer_size = 2 ** 19

counter = 0

#Take data from the PLUTO
data_array = sdr.rx()

#Create spectrogram
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
shifted_fft= np.abs(np.fft.fftshift(np.fft.fft((data_array))))
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

#Plot time domain and spectrogram

plt.subplot(2,1,1)
plt.plot(time_domain, out_data)
plt.xlabel("Time [sec]")
plt.ylabel("Magnitude")

plt.subplot(2,1,2)
plt.pcolormesh(f, t, Sxx, shading="gouraud")
plt.xlabel("Freqeuency [Hz]")
plt.ylabel("Time [sec]")
plt.show()

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


#PED Attempt 1
ideal = dataI + 1j*dataQ
#euclidean_distance = scipy.integrate((ideal - out_data) ** 2)
#Calculate phase of ideal and real data
phase_ideal = np.angle(ideal)

#Initialize arrays
phase_real = np.angle(data_array)
phase_error = np.zeros(len(data_array))

#Calculate phase error by taking the difference between real and ideal phases
for i in range(len(data_array)):
    phase_error[i] = phase_ideal[i] - phase_real[i]
    #print(phase_error)

#plt.scatter(np.real(phase_error), np.imag(phase_error), marker = 'x')
#plt.scatter(np.real(samples_shifted), np.imag(samples_shifted), marker = '.')
#plt.scatter(np.real(samples_of_dpll), np.imag(samples_of_dpll), marker = 'x', color = 'r')
#plt.title('IQ Samples')
#plt.show()

#plt.scatter(time_domain, np.angle(samples_of_dpll),marker = 'x')
#plt.title('Time Domain vs. Phase')
#plt.show()

#Calculate phase of ideal and real data
ideal = dataI + 1j*dataQ
phase_ideal = np.angle(ideal)
phase_real = np.angle(data_array)

#Initialize arrays
phase_error_2 = np.zeros(len(data_array))
data_corrected = np.zeros(len(samples_shifted))

#Calculate phase error by multiplying collected data with reference data
for i in range(len(data_array)):
    phase_error_2[i] = phase_ideal[i] * phase_real[i]
    data_corrected[i] = np.exp(-1j*2*np.pi*phase_error[i]) * samples_shifted[i]
    #print(phase_error)

fft_phase_error_2 = np.fft.fft(phase_error_2)
#plt.plot(fft_phase_error_2)

#Plot IQ data of phase error
plt.scatter(np.real(samples_shifted), np.imag(samples_shifted))
plt.scatter(np.real(phase_error_2), np.imag(phase_error_2), marker = 'x', color = 'r')
plt.show()

#Plot IQ data of corrected phase


#LPF
fc_loop_filter = 20000
loop_filter = scipy.signal.butter(2, fc_loop_filter, btype = 'low', analog = False, output = 'ba',fs=sample_rate)
zi = (2, 6)
#filtered_DPLL = scipy.signal.sosfilt(loop_filter, fft_phase_error_2, axis = '0', zi=zi)
#plt.plot(filtered_DPLL)
#plt.show()
#NCO

#Bit Mapping
#binary_data = ""

#if change_in_phase == 0:
#    binary_data.append("0")
# change_in_phase == pi:
#    binary_data.append("1")
#print(binary_data)

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
#plt.figure(figsize=(9, 5))
#plt.plot(dataI,dataQ)
#plt.xlabel('Inphase')
#plt.ylabel('Quadrature')
#plt.show()

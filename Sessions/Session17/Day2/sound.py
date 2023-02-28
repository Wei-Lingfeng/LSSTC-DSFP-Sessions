import numpy as np
import pydub
import matplotlib.pyplot as plt

# Load mp3 file using pydub
audio = pydub.AudioSegment.from_file("john-denver-take-me-home-country-roads-audio.mp3")

# Extract audio data as numpy array
audio_data = np.array(audio.get_array_of_samples())

# Convert audio data to mono if stereo
if audio.channels > 1:
    audio_data = audio_data.reshape((-1, audio.channels)).mean(axis=1)

# Calculate sampling rate of audio
sampling_rate = audio.frame_rate

# Calculate the length of the audio file in seconds
length = len(audio_data) / sampling_rate

# Define the time window in seconds
window_size = 0.1
step_size = 0.05

# Initialize empty array for frequency spectrum
freq_time_series = []

# Loop through audio data in windows and perform Fourier transform
for i in range(0, len(audio_data) - int(window_size * sampling_rate), int(step_size * sampling_rate)):
    # Extract audio window
    audio_window = audio_data[i:i + int(window_size * sampling_rate)]
    
    # Perform Fourier transform
    freq_data = np.fft.fft(audio_window)
    
    # Append frequency spectrum to time series
    freq_time_series.append(np.abs(freq_data))
    
# Convert time series to numpy array
freq_time_series = np.array(freq_time_series).T

# Create frequency axis in Hz
freq_axis = np.fft.fftfreq(len(audio_window), d=1/sampling_rate)

# Create time axis in seconds
time_axis = np.arange(0, length - window_size, step_size)

# Plot frequency time series
plt.pcolormesh(time_axis, freq_axis, freq_time_series)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency Time Series of Music File")
plt.show()

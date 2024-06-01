import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from memory_profiler import memory_usage

"""
 * ***** Einlesen der Audio-Datei ***** *
"""


def read_wav_file(filename):
    rate, data = wav.read(filename)
    return rate, data


""" 
 * ***** Schnellen Fourieranalys mit Rückgabe positiver Werte ***** *
"""


def perform_fft_pos(data, block_size, sample_rate):
    num_blocks = len(data) // block_size
    frequencies = np.fft.fftfreq(block_size, 1 / sample_rate)
    positive_frequencies = frequencies[:block_size // 2]
    fft_blocks = []

    for i in range(num_blocks):
        block_data = data[i * block_size:(i + 1) * block_size]
        fft_block = fft(block_data)
        fft_blocks.append(np.abs(fft_block)[:block_size // 2])

    return np.array(fft_blocks), positive_frequencies


""" 
 * ***** Schnellen Fourieranalys mit Rückgabe aller Werte ***** *
"""


def perform_fft(data, block_size, sample_rate):
    num_blocks = len(data) // block_size
    frequencies = np.fft.fftfreq(block_size, 1 / sample_rate)
    fft_blocks = []

    for i in range(num_blocks):
        block_data = data[i * block_size:(i + 1) * block_size]
        fft_block = fft(block_data)
        fft_blocks.append(np.abs(fft_block))

    return np.array(fft_blocks), frequencies


""" 
 * ***** Plotten der Daten als Spektraldiagramm  ***** *
"""


def plot_spectrogram(fft_blocks, frequencies, sample_rate, block_size):
    time_bins = np.arange(fft_blocks.shape[0]) * (block_size / sample_rate)
    plt.imshow(20 * np.log10(fft_blocks.T + 1e-10),
               extent=[time_bins.min(), time_bins.max(), frequencies.min(), frequencies.max()], aspect='auto',
               cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()


""" 
 * ***** Plotten der Daten als Frequenz & Amplituden Diagramm ***** *
"""


def plot_frequencies_amplitudes(fft_blocks, frequencies):
    avg_magnitudes = np.mean(fft_blocks, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, avg_magnitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Amplitude Spectrum')
    plt.grid(True)
    plt.show()


""" 
 * ***** Speicherverbrauch plotten ***** *
"""


def plot_memory_usage(memory_data):
    plt.figure(figsize=(10, 6))
    plt.plot(memory_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True)
    plt.show()


""" 
 * ***** Main Methode mit folgenden Argumenten ***** *
 * ***** f gibt den Pfad zur Audio-Datei an
 * ***** --b gibt die Anzahl der Blockgröße
 * ***** --o gibt die Formatierung des Diagramms
 * ***** --p gibt das Vorzeichen der Auswertung an
"""


def main():
    parser = argparse.ArgumentParser(description='Fourier Analysis of WAV file')
    parser.add_argument('f', type=str, help='Path to the WAV file')
    parser.add_argument('--b', type=int, default=1024, help='Block size for FFT')
    parser.add_argument('--o', type=str, choices=['s', 'f'], default='spectrogram',
                        help='Type of output: spectrogram or frequencies')
    parser.add_argument('--p', type=str, choices=['0', '1'], default='1',
                        help='Type of output: only positive or all frequencies ')

    args = parser.parse_args()

    sample_rate, data = read_wav_file(args.f)
    if data.ndim > 1:
        data = data[:, 0]  # Take only one channel if stereo

    if args.p == "1":
        memory_data = memory_usage((perform_fft_pos, (data, args.b, sample_rate)), interval=0.1)
        fft_blocks, frequencies = perform_fft_pos(data, args.b, sample_rate)
    else:
        memory_data = memory_usage((perform_fft, (data, args.b, sample_rate)), interval=0.1)
        fft_blocks, frequencies = perform_fft(data, args.b, sample_rate)

    if args.o == 's':
        plot_spectrogram(fft_blocks, frequencies, sample_rate, args.b)
    elif args.o == 'f':
        plot_frequencies_amplitudes(fft_blocks, frequencies)

    plot_memory_usage(memory_data)


if __name__ == "__main__":
    main()

import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from scipy.fftpack import fft
from multiprocessing import Pool, cpu_count

"""
 * ***** Einlesen der Audio-Datei ***** *
"""


def read_wav_file(filename):
    rate, data = wav.read(filename)
    return rate, data


"""
 * ***** Fouriertransformation für einen Datenblock ***** *
"""


def perform_fft_block(block_data, sample_rate):
    fft_block = fft(block_data)
    return np.abs(fft_block)[:len(fft_block) // 2]


"""
 * ***** Fouriertransformation mit Versatz und Parallelverarbeitung ***** *
"""


def perform_fft(data, block_size, offset, sample_rate):
    num_blocks = (len(data) - block_size) // offset + 1
    blocks = [data[i * offset:i * offset + block_size] for i in range(num_blocks)]

    with Pool(cpu_count()) as pool:
        fft_blocks = pool.starmap(perform_fft_block, [(block, sample_rate) for block in blocks])

    frequencies = np.fft.fftfreq(block_size, 1 / sample_rate)[:block_size // 2]

    return np.array(fft_blocks), frequencies


"""
 * ***** Amplitudenmittelwerte berechnen und ausgeben ***** *
"""


def print_mean_amplitudes(fft_blocks, frequencies, threshold):
    avg_magnitudes = np.mean(fft_blocks, axis=0)

    print("Frequencies with average amplitude above the threshold:")
    for freq, magnitude in zip(frequencies, avg_magnitudes):
        if magnitude > threshold:
            print(f"Frequency: {freq:.2f} Hz, Amplitude: {magnitude:.2f}")

    return avg_magnitudes


"""
 * ***** Plotten der Amplitudenmittelwerte ***** *
"""


def plot_mean_amplitudes(frequencies, avg_magnitudes, threshold):
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, avg_magnitudes, label='Average Amplitudes')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Amplitude Spectrum')
    plt.legend()
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
 * ***** --o gibt den Versatz zwischen den Blöcken
 * ***** --t gibt den Schwellwert für den Amplitudenmittelwert an
"""


def main():
    parser = argparse.ArgumentParser(description='Fourier Analysis of WAV file with block size and offset')
    parser.add_argument('f', type=str, help='Path to the WAV file')
    parser.add_argument('--b', type=int, default=1024, help='Block size for FFT (between 64 and 512)')
    parser.add_argument('--o', type=int, default=1, help='Offset between blocks (between 1 and block size)')
    parser.add_argument('--t', type=float, default=0.0, help='Threshold for average amplitude')

    args = parser.parse_args()

    if not (64 <= args.b <= 512):
        raise ValueError("Block size must be between 64 and 512")
    if not (1 <= args.o <= args.b):
        raise ValueError("Offset must be between 1 and block size")

    sample_rate, data = read_wav_file(args.f)
    if data.ndim > 1:
        data = data[:, 0]  # Take only one channel if stereo

    memory_data = memory_usage((perform_fft, (data, args.b, args.o, sample_rate)), interval=0.1)
    fft_blocks, frequencies = perform_fft(data, args.b, args.o, sample_rate)

    avg_magnitudes = print_mean_amplitudes(fft_blocks, frequencies, args.t)
    plot_mean_amplitudes(frequencies, avg_magnitudes, args.t)

    plot_memory_usage(memory_data)


if __name__ == "__main__":
    main()

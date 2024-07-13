import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pyopencl as cl
from memory_profiler import memory_usage

"""
 * ***** Einlesen der Audio-Datei ***** *
"""


def read_wav_file(filename):
    rate, data = wav.read(filename)
    return rate, data


"""
 * ***** Schnelle Fouriertransformation auf der GPU mit OpenCL ***** *
"""


def perform_fft_gpu(context, queue, data, block_size, offset, sample_rate):
    mf = cl.mem_flags
    data = np.array(data, dtype=np.float32)
    num_blocks = (len(data) - block_size) // offset + 1
    frequencies = np.fft.fftfreq(block_size, 1 / sample_rate)
    positive_frequencies = frequencies[:block_size // 2]

    fft_blocks = np.empty((num_blocks, block_size // 2), dtype=np.float32)

    program_src = """
    __kernel void fft(__global const float *data, __global float *fft_blocks, int block_size, int offset, int num_blocks) {
        int i = get_global_id(0);
        if (i < num_blocks) {
            for (int j = 0; j < block_size / 2; j++) {
                float real = 0.0;
                float imag = 0.0;
                for (int k = 0; k < block_size; k++) {
                    float angle = -2.0f * 3.141592653589793f * j * k / block_size;
                    real += data[i * offset + k] * cos(angle);
                    imag += data[i * offset + k] * sin(angle);
                }
                fft_blocks[i * (block_size / 2) + j] = sqrt(real * real + imag * imag);
            }
        }
    }
    """

    program = cl.Program(context, program_src)
    try:
        program.build()
    except Exception as e:
        print("Error during compilation of OpenCL code:")
        print(e)
        print(program.get_build_info(context.devices[0], cl.program_build_info.LOG))
        raise

    data_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    fft_buffer = cl.Buffer(context, mf.WRITE_ONLY, fft_blocks.nbytes)

    program.fft(queue, (num_blocks,), None, data_buffer, fft_buffer, np.int32(block_size), np.int32(offset), np.int32(num_blocks))
    cl.enqueue_copy(queue, fft_blocks, fft_buffer).wait()

    return fft_blocks, positive_frequencies


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
 * ***** --o gibt den Versatz zwischen den Blöcken an
 * ***** --t gibt den Schwellwert für den Amplitudenmittelwert an
"""


def get_amd_platform():
    platforms = cl.get_platforms()
    for platform in platforms:
        if 'AMD' in platform.name:
            return platform
    raise RuntimeError('AMD platform not found.')


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

    # Get AMD platform and create context and queue
    platform = get_amd_platform()
    context = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platform)])
    queue = cl.CommandQueue(context)

    memory_data = memory_usage((perform_fft_gpu, (context, queue, data, args.b, args.o, sample_rate)), interval=0.1)
    fft_blocks, frequencies = perform_fft_gpu(context, queue, data, args.b, args.o, sample_rate)

    avg_magnitudes = print_mean_amplitudes(fft_blocks, frequencies, args.t)
    plot_mean_amplitudes(frequencies, avg_magnitudes, args.t)

    plot_memory_usage(memory_data)


if __name__ == "__main__":
    main()

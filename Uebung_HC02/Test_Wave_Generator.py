import numpy as np
import wave
import struct


def generate_sine_wave(frequency, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waves = amplitude * np.sin(2 * np.pi * frequency * t)
    return waves


def generate_silence(duration, sample_rate):
    waves = np.zeros(int(sample_rate * duration))
    return waves


def generate_white_noise(duration, sample_rate, amplitude):
    waves = amplitude * np.random.uniform(-1, 1, int(sample_rate * duration))
    return waves


def save_wave(filename, wave_data, sample_rate, num_channels):
    with wave.open(filename, 'w') as wav_file:
        n_frames = len(wave_data)
        wav_file.setparams((num_channels, 2, sample_rate, n_frames, 'NONE', 'not compressed'))
        for sample in wave_data:
            wav_file.writeframes(struct.pack('h', int(sample * 32767)))


def generate_test_wav_files():
    sample_rate = 44100
    amplitude = 0.5

    # Test 1: Short 440 Hz Sine Wave (10 second)
    sine_wave_440_short = generate_sine_wave(440, 10.0, sample_rate, amplitude)
    save_wave('Wav_Files/test_440hz_10s.wav', sine_wave_440_short, sample_rate, 1)

    # Test 2: Long 440 Hz Sine Wave (100 seconds)
    sine_wave_440_long = generate_sine_wave(440, 100.0, sample_rate, amplitude)
    save_wave('Wav_Files/test_440hz_100s.wav', sine_wave_440_long, sample_rate, 1)

    # Test 3: White Noise (300 seconds)
    white_noise = generate_white_noise(300.0, sample_rate, amplitude)
    save_wave('Wav_Files/test_white_noise_300s.wav', white_noise, sample_rate, 1)

    # Test 4: Silence (150 seconds)
    silence = generate_silence(150.0, sample_rate)
    save_wave('Wav_Files/test_silence_150s.wav', silence, sample_rate, 1)

    print("WAV Test files generated successfully.")


generate_test_wav_files()

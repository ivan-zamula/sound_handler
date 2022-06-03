"""Основное файл приложения"""
import os
from app.classes import Audio_Signal
import librosa

FILENAME = 'audio/gamma2.wav'


if __name__ == "__main__":
    # fn_wav = os.path.join('', '', '', FILENAME)
    print(os.getcwd())
    # fn_wav = os.path.join('', '', '', 'exm_1.wav')  # c_dur.wav
    # clip, sample_rate = librosa.load(fn_wav, sr=14700)

    audio_signal = Audio_Signal(FILENAME)
    audio_signal.draw_amplitude_time_plot()
    audio_signal.make_fft()
    audio_signal.shift_fft()
    audio_signal.get_positive_spectr()
    audio_signal.set_window()
    audio_signal.get_welch()
    audio_signal.find_peaks()
    audio_signal.print_notes()

    # a_utils = AudioHandler(audio)
    # a_utils.draw_amplitude_time_plot()


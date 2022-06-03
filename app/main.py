"""Основное файл приложения"""
import os
from app.classes import Audio_Signal, KeyFinder
import librosa

FILENAME = 'audio/in_forest.wav'


def load(audio_path):
    """Загрузка файла"""
    # return a2n.audio_from_file(self.audio_path)
    return librosa.load(audio_path)


def get_hpss_harmonic(signal):
    """"Разделение на гармоническую и ударную составляющие.
    В дальнейшем, обработка идет по гармоническим составляющим"""
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    return y_harmonic


if __name__ == "__main__":
    signal, sample_rate = load('audio/gamma3.wav')

    audio_signal = Audio_Signal(signal, sample_rate)
    audio_signal.audio_handler()

    signal, sample_rate = load(FILENAME)
    signal = get_hpss_harmonic(signal)
    key_finder = KeyFinder(signal, sample_rate, time_to=22)
    key_finder.print_spectr()
    key_finder.corr_table()
    key_finder.print_key()
    key_finder.plot_spectrogram("Спектрограмма сигнала")


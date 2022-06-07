"""Основное файл приложения"""
import os
from app.classes import AudioHandler, KeyFinder
import librosa

FILENAME = 'audio/in_forest.wav'


def load(audio_path):
    """Загрузка файла"""
    return librosa.load(audio_path)


def get_hpss_harmonic(signal):
    """"Разделение на гармоническую и ударную составляющие.
    В дальнейшем, обработка идет по гармоническим составляющим"""
    signal_harmonic, signal_percussive = librosa.effects.hpss(signal)
    return signal_harmonic


if __name__ == "__main__":
    signal, sample_rate = load('audio/gamma3.wav')

    audio_signal = AudioHandler(signal, sample_rate)
    audio_signal.audio_handler_run()

    signal, sample_rate = load(FILENAME)
    signal = get_hpss_harmonic(signal)

    key_finder = KeyFinder(signal, sample_rate, time_to=22)
    key_finder.key_finder_run()



"""Описание классов"""
import librosa


class AudioSignal():
    """Класс сигнала: содержит аттрибуты и методы для хранения и обработки основных характеристик сигнала"""

    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.signal, self.sample_rate = self.load(self)

        pass

    def load(self):
        return librosa.load(self.audio_path)

    def hpss(self):
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        pass


class AudioHandler():
    """Обработчик сигнала: основные функции предварительной обработки сигнала"""

    def __init__(self):
        pass


class KeyFinder():
    """Поиск тональности: функции для поиска тональности сигнала"""

    def __init__(self):
        pass

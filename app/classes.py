"""Описание классов"""
import librosa
import librosa.display
import numpy as np
import peakutils

from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftshift
from scipy.signal import get_window, welch
import audio2numpy as a2n


class Audio_Signal():
    """Класс сигнала: содержит аттрибуты и методы для хранения и обработки основных характеристик сигнала
    Предназначен для поиска нот"""

    def __init__(self, signal, sample_rate):
        self.signal = signal
        self.sample_rate = sample_rate
        self.time_steps = np.arange(len(self.signal)) / self.sample_rate
        self.n_fft = len(self.signal)
        self.X = None
        self.win = None
        self.wx = None
        self.freq = None
        self.power = None

    def load(self):
        """Загрузка файла"""
        # return a2n.audio_from_file(self.audio_path)
        return librosa.load(self.audio_path)

    # Метод для отображения графика Амплитуда/Время
    def draw_amplitude_time_plot(self):
        """График амплитуда/время"""
        fig, ax = plt.subplots(2, figsize=(12, 5))
        fig.subplots_adjust(hspace=0.5)

        # Отображение всего сигнала на графике
        ax[0].plot(self.time_steps, self.signal)
        ax[0].set_xlabel('Время (сек.)')
        ax[0].set_ylabel('Амплитуда')
        ax[0].set_title('Сигнал: {} ({} сэмплов)'.format('сигнал во временной области', self.n_fft))

        start = 1
        ax[0].axvline(start / self.sample_rate, c='r')
        ax[0].axvline((start + self.n_fft) / self.sample_rate, c='r')

        # Отображение N отрезков
        ax[1].plot(self.signal)
        ax[1].set_xlabel('Сэмплы')
        ax[1].set_ylabel('Амплитуда')
        ax[1].set_title('Сигнал: {} ({} сэмплов)'.format('сигнал во временной области', self.n_fft));
        plt.show()

    # Метод для выполнения быстрого преобразования Фурье
    def make_fft(self):
        """Преобразование Фурье"""
        self.X = fft(self.signal, self.n_fft)
        plt.plot(abs(self.X))
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.title('Быстрое преобразование Фурье')
        plt.show()

    # Метод для преобразования спектра и перемещения отчетов симметрично, относительно нулевой частоты
    def shift_fft(self):
        """Симметричное смещение относительно нулевой частоты"""
        Xshift = fftshift(self.X)
        F = np.linspace(-1 * self.sample_rate / 2, self.sample_rate / 2, len(self.X));
        plt.plot(F, abs(Xshift))
        plt.xlim(-2000, 2000)
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.title('Преобразование спектра и симметричный сдвиг спектра, относительно нулевой частоты')
        plt.show()

    # Метод формирования спектра в области положительных частот
    def get_positive_spectr(self):
        """Получение спектра в области положительных частот"""
        Xpositive = abs(self.X // 2)
        Fpositive = np.linspace(0, self.sample_rate / 2, len(Xpositive))
        plt.xlim(0, 1500)
        plt.plot(Fpositive, abs(Xpositive))
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.title('Вывод спектра в области положительных частот')
        plt.show()

    # Метод применения оконной функции
    def set_window(self):
        """Применение оконной функции"""
        self.win = get_window('rect', self.n_fft)
        # Варианты оконных функций hann #rect #triang #hamm
        self.wx = self.signal * self.win

    # Метод оценки спектральной плотности мощности сигнала
    def get_welch(self):
        """Оценка спектральной мощности сигнала"""
        self.freq, self.power = welch(self.wx,
                                      fs=self.sample_rate,
                                      window=self.win,
                                      noverlap=100,
                                      nfft=self.n_fft,
                                      axis=0,
                                      scaling='density',
                                      detrend=False)

        plt.xlim(0, 1200)
        plt.plot(self.freq, self.power)
        return self.freq, self.power

    # Метод формирования списка найденных нот
    def find_notes(self):
        notes_list = []
        notes = []
        for i in self.freq[self.index]:
            notes_list.append(librosa.hz_to_note(i, octave=True))
        # Отберем уникальные значения
        notes_list = list(set(notes_list))
        for i in notes_list:
            notes.append(librosa.note_to_hz(i))
        return notes, notes_list

    # Метод для нахождения максимумов (пиков) спектра
    def find_peaks(self):
        self.index = peakutils.indexes(self.power, thres=0.100, min_dist=10)
        plt.hlines(0.00001, 0, 1200, colors="green", linestyles="dotted", label="Нижний порог")
        plt.plot(self.freq, self.power, lw=0.8, alpha=0.4, color="red")
        plt.plot(self.freq[self.index], self.power[self.index], marker="o", ls="", ms=3)

        # найдем ноты и выведем уникальные значения
        notes, notes_list = self.find_notes()
        # Отобразим значение найденных нот
        plt.xticks(notes, notes_list, rotation='vertical')

        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')
        plt.title('Нахождение пиков спектра')
        plt.legend()

        plt.show()

    # Метод печати нот
    def print_notes(self):
        # print(self.f[self.index])
        notes = []
        for i in self.freq[self.index]:
            s = librosa.hz_to_note(i, octave=True, cents=False)
            if s not in notes:
                notes.append(s)

        for i in notes:
            print(i)
            # print(librosa.hz_to_note(i, octave=True, cents=False))

    def audio_handler(self):
        self.draw_amplitude_time_plot()
        self.make_fft()
        self.shift_fft()
        self.get_positive_spectr()
        self.set_window()
        self.get_welch()
        self.find_peaks()
        self.print_notes()


class KeyFinder(object):
    """Поиск тональности: функции для поиска тональности сигнала
    Предназначен для поиска тональности"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # использование механизма поиска тональностей Крумхансла-Шмуклера -
    # который сравнивает приведенные выще данные цветности с ключевыми профилями
    # мажорной и минорной тональностей
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    ALT_KEY_PROBABILITY = 0.7

    def __init__(self, signal, sample_rate, time_from=None, time_to=None):
        # Инициализация аттрибутов класса
        self.signal = signal
        self.sample_rate = sample_rate
        self.time_from = time_from
        self.time_to = time_to
        self.y_segment = self.get_part_of_signal()
        self.spectrogram = librosa.feature.chroma_stft(y=signal, sr=sample_rate, n_chroma=12, n_fft=2048)

        # spectr_values мощность нот сигнала на spectrogram в заданном временном интервале
        self.spectr_values = []
        for i in range(12):
            self.spectr_values.append(np.sum(self.spectrogram[i]))
        # notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # словарь связывающий название ноты с соответствующей мощностью (интенсивностью в песне)
        self.dict_notes_freq = {self.notes[i]: self.spectr_values[i] for i in range(12)}
        # {'C':100, 'C#': 34, }

        # список тональностей
        keys = [self.notes[i] + ' major' for i in range(12)] + [self.notes[i] + ' minor' for i in range(12)]

        # находим сходство между мощностью ноты во временном интервале и ключевым профилем начиная с каждого из 12
        # основных нот и создаем словарь тональностей с учетом сходства (корреляции)
        self.dict_minor_corrs = []
        self.dict_major_corrs = []
        for i in range(12):
            key_test = [self.dict_notes_freq.get(self.notes[(i + m) % 12]) for m in range(12)]
            # 0, 1 ..12 -> 0,1,...11
            # 1,2 ..13 -> 1,2..11,0
            # correlation coefficients (strengths of correlation for each key)
            # коэф. корреляции
            self.dict_major_corrs.append(round(np.corrcoef(self.major_profile, key_test)[1, 0], 3))
            self.dict_minor_corrs.append(round(np.corrcoef(self.minor_profile, key_test)[1, 0], 3))

        # формирует словарь с названием тональностей (ключ) и соответствующим списком коэфф. корреляции
        self.dict_keys = {**{keys[i]: self.dict_major_corrs[i] for i in range(12)},
                         **{keys[i + 12]: self.dict_minor_corrs[i] for i in range(12)}}

        # this attribute represents the key determined by the algorithm
        # key - самая подход тональность (название)
        # значение коэф.корр - максимальное
        self.key = max(self.dict_keys, key=self.dict_keys.get)
        self.best_corr = max(self.dict_keys.values())

        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        # в цикле проходит по всему словарю и находит вторые значения больше чем 90% от лучшего
        self.alt_key = None
        self.alt_best_corr = 0

        for key, corr in self.dict_keys.items():
            if corr > self.best_corr * self.ALT_KEY_PROBABILITY and corr != self.best_corr and corr > self.alt_best_corr:
                self.alt_key = key
                self.alt_best_corr = corr

    # получение части сигнала
    def get_part_of_signal(self):
        if self.time_from is not None:
            self.time_from = librosa.time_to_samples(self.time_from, sr=self.sample_rate)
        if self.time_to is not None:
            self.time_to = librosa.time_to_samples(self.time_to, sr=self.sample_rate)
        return self.signal[self.time_from:self.time_to]

    # печать отношения значения каждой мощности ноты к максимальному значению мощности
    def print_spectr(self):
        spectr_max = max(self.spectr_values)
        for key, spectr in self.dict_notes_freq.items():
            print(key, '\t', f'{spectr / spectr_max:5.3f}')

    # печать таблицы коэффициентов корреляции для каждой тональности
    def corr_table(self):
        for key, corr in self.dict_keys.items():
            print(key, '\t', f'{corr:6.3f}')

    # печать лучшего в тональность совпадения и лучшего из всех тональностей, а также альтернативные
    def print_key(self):
        print("Наиболее вероятная тональность: ", max(self.dict_keys, key=self.dict_keys.get), ", корреляция: ", self.best_corr, sep='')
        if self.alt_key is not None:
            print("Альтернативаная тональность: ", self.alt_key, ", корреляция: ", self.alt_best_corr, sep='')

    # печать хромограммы
    def plot_spectrogram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.signal, sr=self.sample_rate, bins_per_octave=24)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(C, sr=self.sample_rate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title("Спектрограмма сигнала")
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

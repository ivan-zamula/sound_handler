# Musical Note and Key Analysis (2022)

[English](README.md) | [Русский](README.ru.md)

Python experiments in audio signal analysis using the fast Fourier transform (FFT). The code finds dominant frequencies and musical notes, and estimates the key of an audio signal.

## Methods and tools

- Waveform and spectrum plots.
- FFT, window functions and Welch power spectral density.
- Spectral peak detection and conversion from frequencies to notes.
- Key estimation from chroma features and major/minor profile correlations.

Built with librosa, NumPy, SciPy, Matplotlib and PeakUtils. Dependency versions are listed in [requirements.txt](requirements.txt).

## Project files

| Path | Purpose |
| --- | --- |
| [app/classes.py](app/classes.py) | `AudioHandler` for signal analysis and `KeyFinder` for key estimation |
| [app/main.py](app/main.py) | Audio loading and analysis examples |
| [app/audio](app/audio) | Three audio samples used by the experiments |

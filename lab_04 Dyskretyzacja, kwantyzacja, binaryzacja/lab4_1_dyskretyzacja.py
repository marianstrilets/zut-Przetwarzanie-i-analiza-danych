import numpy as np
import matplotlib.pyplot as plt

# 1. Przygotuj funkcję generującą zdyskretyzowany sygnał sinus na podstawie parametrów:
#       Częstotliwość sygnału f i Częstotliwość próbkowania Fs
# 2. Funkcja powinna zwracać tablice:
#       t - tablicę ”czasu” (zakres od 0 do 1 z krokiem równym częstości próbkowania: Ts = 1 / Fs)
#       s - wartości wygenerowanego sygnału sinus: s (t) = sin (2πft)
# 3. Wyświetl wykresy zawierające sygnał o częstotliwości 10Hz za pomocą próbkowania:
#       20Hz 21Hz 30Hz 45Hz 50Hz 100Hz 150Hz 200Hz 250Hz 1000Hz


def gen_discret_sygnal_sin(f, Fs):
    Tc = 1          # Tc - [s] czas trwania (w sekundach)
    Ts = 1 / Fs     # Ts - okres probkowania

    # t - tablicę ”czasu” (zakres od 0 do 1 z krokiem równym częstości próbkowania: Ts = 1 / Fs)
    t = np.arange(0, Tc, Ts)

    # s - wartości wygenerowanego sygnału sinus: s(t) = sin(2πft)
    s = np.sin(2 * np.pi * f * t)
    return t, s


f = 10  # f - [Hz] Częstotliwość sygnału

# Fs- [Hz] Częstotliwość próbkowania
Fs = [20, 21, 30, 45, 50, 100, 150, 200, 250, 1000]

for Fs_i in Fs:
    t, s = gen_discret_sygnal_sin(f, Fs_i)
    plt.plot(t, s)
    plt.title(
        f'Sygnał sinusoidalny, częstotliwość sygnału {f} Hz, próbkowania {Fs_i} Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda')
    plt.show()

# 4. Czy istnieje twierdzenie, które określa z jaką częstotliwością należy próbkować, aby móc wiernie odtworzyć sygnał? Jak się nazywa?
# Odp:
#       Tak, istnieje takie twierdzenie. Nazywa się ono twierdzeniem o próbkowaniu Nyquista-Shannona.
#       Mówi ono, że aby móc dokładnie odtworzyć sygnał ciągły za pomocą jego próbek,
#       częstotliwość próbkowania musi być co najmniej dwukrotnie większa niż najwyższa częstotliwość składowa sygnału ciągłego.
#       W przeciwnym przypadku może wystąpić zjawisko aliasingu, które zakłóca wynik i uniemożliwia dokładne odtworzenie sygnału.
# 5. Jak nazywa się zjawisko, które z powodu błędnie dobranej częstotliwości próbkowania powoduje błędną interpretację sygnału?
# Odp:
#       Zjawisko to nazywa się aliasing. Aliasowanie występuje, gdy częstotliwość próbkowania jest zbyt niska w stosunku do
#       częstotliwości sygnału, co powoduje, że wyższe częstotliwości w sygnale są nieprawidłowo reprezentowane jako niższe
#       częstotliwości w próbkach. Efektem tego jest błędna interpretacja sygnału i zniekształcenie jego kształtu.
# 6.  Znajdź za pomocą internetu obraz, na którym występuje wskazane zjawisko (przy wyświetlaniu go w znacznie mniejszej rozdzielczości).
# Odp:
#       Aliasing – nieodwracalne zniekształcenie sygnału w procesie próbkowania wynikające z niespełnienia założeń twierdzenia o próbkowaniu.
#       Zniekształcenie objawia się obecnością w wynikowym sygnale składowych o błędnych częstotliwościach (aliasów)
#       https://pl.wikipedia.org/wiki/Aliasing_%28przetwarzanie_sygna%C5%82%C3%B3w%29#/media/Plik:AliasingSines.svg
#       Przykład dwóch różnych sinusoid pasujących do tego samego wzoru próbek.
#       W rzeczywistości jest nieskończenie wiele takich sinusoid, które przechodzą przez ten sam zbiór punktów

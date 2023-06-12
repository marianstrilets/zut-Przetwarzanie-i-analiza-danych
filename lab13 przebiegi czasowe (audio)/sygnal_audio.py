#======================================================================================================
#   Laboratorium – ﬁltracja przebiegów czasowych (audio), zastosowanie okien kroczących, FFT  
#======================================================================================================
import sounddevice as sd
import soundﬁle as sf
import numpy as np
import matplotlib.pyplot as plt
#======================================================================================================
#==================================>>> 1 Sygnał audio <<<==============================================
#======================================================================================================
#   1. Korzystając z mikrofonu w komputerze lub dyktafonu w smartfonie 
#   nagraj sekwencję audio ”Jestem studentem informatyki”.
#   2. Wczytaj nagrany plik dźwiękowy do zmiennej s. Odczytaj/oblicz podstawowe parametry sygnału 
#   (czas trwania, częstotliwość próbkowania, rozdzielczość bitowa, liczba kanałów). 
#======================================================================================================

#    Wczytahie nagranego pliku dźwiękowego
s, fs = sf.read('audio.wav', dtype='float32')
# s  - to tablica zawierająca wartości próbek sygnału, każda z jej kolumn zawiera osobny
#      kanał wartości (lewy i prawy), chyba że sygnał jest monofoniczny; 
# fs - to częstotliwość próbkowania, określa liczbę próbek na sekundę;

#   Odzytanie nagranego pliku dźwiękowego
sd.play(s, fs)    # rozpoczęcie odtwarzania;
print('\n---------------->>>>> audio.wav <<<<<----------------\n')
status = sd.wait()# odczekanie z wykonaniem kolejnych instrukcji do momentu, aż dźwięk skończy się odtwarzać.

# Czas trwania
calc_duration = len(s) / fs

# Rozdzielczośc bitowa
bit_depth = 16 

# Liczba kanałów
num_channels = s.shape[1] if len(s.shape) > 1 else 1

# Wyświetlanie wyników
print("\n\tCzas trwania: {:.2f} sekundy".format(calc_duration))
print("\tCzęstotliwość próbkowania: {} Hz".format(fs))
print("\tRozdzielczość bitowa: {} bitów".format(bit_depth))
print("\tLiczba kanałów: {}".format(num_channels))

#======================================================================================================
#   3. Wyświetlić sygnał tak, aby na osi poziomej znajdowała się jednostka czasu [ms] (konieczność 
#   przeliczenia zakresu). Jeżeli po wczytaNiu sygnał nie jest znormalizowany, należy przeprowadzić 
#   normalizację wartości do zakresu [-1;1]. W przypadku sygnału stereo wykorzystaj dowolny z kanałów: 
#   lewy lub prawy (do końca instrukcji pracujemy wyłącznie na jednym kanale, 
#   jeżeli nasz sygnał jest stereofoniczny).   
#======================================================================================================
# Wybór jednego kanału (dla sygnału stereo)
channel = 0  # 0 dla lewego kanału, 1 dla prawego kanału

# Normalizacja wartości do zakresu [-1, 1]
normalized_signal = s[:, channel] / np.max(np.abs(s[:, channel]))

# Obliczanie osi czasu w jednostkach [ms]
time = np.arange(len(normalized_signal)) * 1000 / fs

# Wyświetlanie sygnału
plt.figure('Sygnał audio')
plt.plot(time, normalized_signal)
plt.xlabel('Czas [ms]')
plt.ylabel('Amplituda')
plt.title('Sygnał audio')
plt.show()

#======================================================================================================
#   4. Sprawdź, czy dynamika sygnału jest odpowiednia? Czy zakres amplitudy jest odpowiednio 
#   wykorzystany? Czy nie występuje przesterowanie? Jaka jest amplituda szumu na początku i na końcu 
#   nagrania? Czy szum ma charakter losowy?   
#   5. Jeżeli występują problemy warto powtórzyć nagranie, zadbać o ciszę w pomieszczeniu, ewentualnie 
#   zmienić mikrofon. 
#   Poprawne nagranie dźwięku ułatwia też odpowiedni program, z darmowych przykładowo Audacity
#======================================================================================================
# Sprawdzenie dynamiki sygnału
max_amplituda = np.max(np.abs(s))
min_amplituda = np.min(np.abs(s))
range_amplitude = max_amplituda - min_amplituda

print("\n\tMaksymalna amplituda:", max_amplituda)
print("\tMinimalna amplituda:", min_amplituda)
print("\tZakres amplitudy:", range_amplitude)
  
# Obliczenie amplitudy szumu na początku i na końcu nagrania
noise_amplituda_begin = np.mean(np.abs(s[:int(fs)]))  # Zakres próbek na początku
noise_amplituda_end   =  np.mean(np.abs(s[-int(fs):]))  # Zakres próbek na końcu

print("\n\tAmplituda szumu na początku nagrania:", noise_amplituda_begin)
print("\tAmplituda szumu na końcu nagrania:   ", noise_amplituda_end)

#======================================================================================================
#==========================>>> 2 Zastosowanie okien kroczących <<<=====================================
#======================================================================================================
#   1. Podzielić sygnał na ramki (okna) długości 10 ms i obliczyć dla każdej
#   ramki dwie statystyki – funkcję energii E oraz funkcję przejść przez zero Z:
#   s – wektor sygnału, j – numer ramki, n – długość ramki w próbkach.
#   Wskazówka 1: W zależności od wartości parametru częstotliwości próbkowania, dziesięciu 
#               milisekundom odpowiada różna liczba próbek.
#   Wskazówka 2: Dla każdej ramki, składającej się z n próbek, wyznaczamy dwie liczby 
#               (energię i liczbę przejść przez 0).
#======================================================================================================






# Funkcja obliczająca energię ramki
def E(frame):
    energy = np.sum(frame ** 2)
    return energy

# Funkcja obliczająca liczbę przejść przez zero w ramce
def Z(frame):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
    return zero_crossings

n = int(0.01 * fs)  # Długość ramki w próbkach (10 ms)
overlap = int(0.5 * n)  # Przesunięcie ramek (50% nakładu)

num_frames = len(s) // (n - overlap)  # Obliczanie liczby ramek
energies = np.zeros(num_frames)
zero_crossings = np.zeros(num_frames)

for j in range(num_frames):
    start = j * (n - overlap)
    end = start + n
    frame = s[start:end]
    energies[j] = E(frame)  # Obliczanie energii dla ramki
    zero_crossings[j] = Z(frame)  # Obliczanie liczby przejść przez zero dla ramki

# Normalizacja funkcji energii i przejść przez zero
normalized_energies = (energies - np.min(energies)) / (np.max(energies) - np.min(energies))
normalized_zero_crossings = (zero_crossings - np.min(zero_crossings)) / (np.max(zero_crossings) - np.min(zero_crossings))

# Tworzenie wektora ramki
frame_index = np.arange(num_frames) * 2

# Wizualizacja funkcji energii i przejść przez zero
plt.figure('Sygnał audio')
plt.plot(frame_index, normalized_energies, color='red')
plt.plot(frame_index, normalized_zero_crossings, color='blue')
plt.xlabel('Numer ramki')
plt.ylabel('Amplituda')
plt.title('Funkcje energii i przejść przez zero')
plt.ylim([0, 1])
plt.legend(['Energia', 'Przejścia przez zero'])
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# 1. Pobierz z internetu obraz przedstawiający obiekt na niejednorodnym tle (np. gradiencie)
# 2. Wczytaj obraz, zamień go na skalę szarości za pomocą jednej z wcześniej stosowanych metod i wygeneruj histogram
# 3. Napisz funkcję, która na podstawie histogramu określi punkt progowania (zazwyczaj
#       lokalne minimum między dwoma ‘klasami‘ kolorów,tak jak na poniższym obrazku’)
# 4. Zbinaryzuj (ustaw wartości na 0 lub 1) wartości pikseli obrazka zgodnie z otrzymanym progiem
# 5. Wyświetl obraz z wysegmentowanym obiektem (segmentacja, czyli usunięcie niepotrzebnych elementów,
#       jak np. tła i uwypukleniu tych ważniejszych)

# Wczytanie obrazka
img = plt.imread('./png_gradient.jpg')


gray_img = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray_img[i, j] = (np.max(img[i, j, :]) + np.min(img[i, j, :]))/2

# Wygenerowanie histogramu
hist, bins = np.histogram(gray_img.ravel(), bins=256, range=(0, 255))

# Znalezienie punktu progowania jako lokalne minimum między dwoma klasami kolorów


def find_threshold(hist):
    n = len(hist)
    maximum = np.max(hist)
    threshold = -1
    for i in range(1, n-1):
        if hist[i-1] < hist[i] and hist[i+1] < hist[i] and hist[i] < maximum:
            maximum = hist[i]
            threshold = i
    return threshold


threshold = find_threshold(hist)
print("Punkt progowania wynosi:", threshold)


hist, bins = exposure.histogram(gray_img)
val = bins[find_threshold(hist)]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
ax1.imshow(gray_img, cmap='gray', interpolation='nearest')
ax2.imshow(gray_img < val, cmap='gray', interpolation='nearest')
ax3.plot(bins, hist, color='gray', lw=2)
ax3.axvline(val, color='blue', linestyle='dashed', linewidth=3)
plt.show()

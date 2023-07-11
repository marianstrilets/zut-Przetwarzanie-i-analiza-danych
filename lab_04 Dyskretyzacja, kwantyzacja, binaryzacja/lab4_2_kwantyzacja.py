import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Wczytaj wcześniej znaleziony obrazek
img = plt.imread('aliasingSin.jpg')
#plt.imshow(img)
#plt.title('Obrazek')
#plt.show()

# 2. Wykonaj polecenie, które zwróci ile wymiarów ma wczytana macierz (obrazek)
img_dims = np.shape(img)
print(
    f"Wymiary obrazka: \n\tilość wierszy: {img_dims[0]} \n\tilość kolumn: {img_dims[1]} \n\tilość kanałów kolorów: {img_dims[2]} \n{img_dims}")

# 3. Wykonaj polecenie, które zwróci iloma wartościami jest opisywany pojedynczy piksel (inaczej:
#       z ilu wartości składa się najgłębszy wymiar)
num_channels = np.shape(img)[-1]
print("\nLiczba kanałów:\n\t", num_channels)

# 4. Przekształć obraz do skali szarości za pomocą 3 różnych metod (zapisz jako 3 różne macierze):
#       • Wyznaczenie jasności piksela: (max(R,G, B) + min(R,G, B))/2
#       • Uśrednienie wartości piksela: (R + G + B)/3
#       • Wyznaczenie luminancji piksela: 0.21R + 0.72G + 0.07B

# Wyznaczenie jasności piksela: (max(R,G, B) + min(R,G, B))/2
gray_img1 = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray_img1[i, j] = (np.max(img[i, j, :]) + np.min(img[i, j, :]))/2
#plt.imshow(gray_img1, cmap='gray')
#plt.title('Jasność piksela')
#plt.show()

# Uśrednienie wartości piksela: (R + G + B)/3
gray_img2 = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray_img2[i, j] = np.mean(img[i, j, :])
#plt.imshow(gray_img2, cmap='gray')
#plt.title('Uśrednienie wartości piksela')
#plt.show()

# Wyznaczenie luminancji piksela: 0.21R + 0.72G + 0.07B
gray_img3 = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        gray_img3[i, j] = 0.21*img[i, j, 0] + \
            0.72*img[i, j, 1] + 0.07*img[i, j, 2]
#plt.imshow(gray_img2, cmap='gray')
#plt.title('Luminancja piksela')
#plt.show()

# 5. Wygeneruj histogram dla każdego z otrzymanych „szarych” obrazów (funkcja histogram z pakietu numpy)
# Wygenerowanie histogramów
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].hist(gray_img1.ravel(), bins=256, color='gray')
axs[0, 0].set_title('Histogram - jasność')
axs[0, 1].hist(gray_img2.ravel(), bins=256, color='gray')
axs[0, 1].set_title('Histogram - uśrednienie')
axs[1, 0].hist(gray_img3.ravel(), bins=256, color='gray')
axs[1, 0].set_title('Histogram - luminancja')

# 6. Dla dowolnego z wygenerowanych obrazów, za pomocą parametru bins zredukuj liczbę kolorów
#       na histogramie do 16 i wyświetl zakresy nowych kolorów
# Kwantyzacja do 16 kolorów
hist, bins = np.histogram(gray_img1, bins=16)
new_colors = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
# Wygenerowanie nowego obrazka ze zredukowaną liczbą kolorów
new_img = np.zeros_like(gray_img1)
for i in range(len(new_colors)):
    new_img[(gray_img1 >= bins[i]) & (gray_img1 < bins[i+1])] = new_colors[i]

# 7. Stwórz kolejną macierz (obrazek) ze zredukowaną liczbą kolorów (jako nową wartość piksela
#       przyjmnij środek przedziału zwróconego przez funkcję histogramu)
# Wygenerowanie histograma dla nowego obrazka
axs[1, 1].hist(new_img.ravel(), bins=16, color='gray')
axs[1, 1].set_title('Histogram - 16 kolorów')


# 8. Wyświetl wszystkie obrazy i ich histogramy
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].imshow(img)
axs[0, 0].set_title('Oryginalny obraz')
axs[0, 1].imshow(gray_img1, cmap='gray')
axs[0, 1].set_title('Jasność')
axs[1, 0].imshow(gray_img2, cmap='gray')
axs[1, 0].set_title('Uśrednienie')
axs[1, 1].imshow(gray_img3, cmap='gray')
axs[1, 1].set_title('Luminancja')

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].imshow(gray_img1, cmap='gray')
axs[0].set_title('Jasność - 16 kolorów')
axs[1].imshow(new_img, cmap='gray')
axs[1].set_title('16 kolorów')

plt.show()
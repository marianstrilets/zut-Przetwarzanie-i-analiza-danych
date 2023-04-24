import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# ====================================================================================================================
# 1. Implementacja PCA i funkcji dwuwymiarowej wizualizacji przestrzeni cech
# ====================================================================================================================
# ---------------------------------------- (a) -----------------------------------------------------------------------
#   (a) wygenerować w sposób losowy zbiór 200 obiektów dwuwymiarowych za pomocą funkcji z numpy dot i rand lub randn
# --------------------------------------------------------------------------------------------------------------------
print('\n--------------- 1 -------------- \n\ta:\n')
# ustawienie seeda dla powtarzalności wyników
np.random.seed(30)

# Generujemy zbiór 200 obiektów dwuwymiarowych za pomocą funkcji dot i rand
rng = np.random.RandomState(1)
objects = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

 # wyświetl pierwsze 5 obiektów
print(objects[:5]) 
print("Ilość obiektów w zbiore dwuwymiarowym: ", objects.shape)
# ---------------------------------------- (b) -----------------------------------------------------------------------
#   (b) zwizualizować obiekty na pomocą funkcji matplotlib, np. scatter
# wykres punktowy przedstawiający obiekty w przestrzeni dwuwymiarowej
# --------------------------------------------------------------------------------------------------------------------
# wykres punktowy
plt.scatter(objects[:, 0], objects[:, 1])
plt.axis('equal')
plt.title('1b: wykres punktowy')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# ---------------------------------------- (c) -----------------------------------------------------------------------
#   (c) dokonać redukcji do jednego wymiaru za pomocą własnej funkcji wiPCA i zwizualizować wektory
#       własne oraz rzut wygenerowanych obiektów na pierwszą składową, w sposób podobny do Rysunku 1
# --------------------------------------------------------------------------------------------------------------------
# Funkcja rysująca wektory:
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=3,
                      shrinkA=0, shrinkB=0, color='b')
    ax.annotate('', v1, v0, arrowprops=arrowprops)
# --------------------------------------------------------------------------------------------------------------------
# Implementacja wiPCA
def wiPCA(X, k=1):
    # Argumenty:
    #   X - macierz danych
    #   k - liczba składowych głównych do zachowania (domyślnie 1)
    # Zwraca:
    #   X_pca - zredukowana macierz danych
    #   W - macierz transformacji
    #   eig_vals - wartości własne macierzy kowariancji
    mean = np.mean(X, axis=0)       # Obliczanie średniej dla każdej kolumny
    X_centered = X - mean           # Odjęcie średniej od każdej kolumny
    cov_mat = np.cov(X_centered.T)  # Obliczenie macierzy kowariancji

    # Obliczenie wartości i wektorów własnych macierzy kowariancji
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Sortowanie wartości własnych w kolejności malejącej
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Wybieranie k największych wartości własnych
    W = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(k)])
    eig_vals = [eig_pairs[i][0] for i in range(k)]

    X_pca = np.dot(X_centered, W)   # Przekształcanie danych
    return X_pca, W, eig_vals       # Zwracanie wyników
# --------------------------------------------------------------------------------------------------------------------
# Wywołanie funkcji PCA original shape: (200, 2)
pca = PCA(n_components=2)
pca.fit(objects)
plt.scatter(objects[:, 0], objects[:, 1], alpha=0.3)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
    
# Wywołanie funkcji PCA transformed shape: (200, 1)
pca = PCA(n_components=1)
pca.fit(objects)
objects_pca = pca.transform(objects)
objects_new = pca.inverse_transform(objects_pca)
plt.scatter(objects_new[:, 0], objects_new[:, 1], alpha=0.3)

print("original shape:   ", objects.shape)
print("transformed shape:", objects_pca.shape)

# Wykres wizualizacja przestrzeni cech i wektorów własnych
plt.axis('equal')
plt.title('1c: Wizualizacja przestrzeni cech i wektorów własnych')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# ====================================================================================================================

# ====================================================================================================================
# 2. Testowanie PCA na zbiorze iris
# ====================================================================================================================
# ---------------------------------------- (a) -----------------------------------------------------------------------
# (a) Wczytać zbiór iris (sposób j.w.)
# --------------------------------------------------------------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target
# ---------------------------------------- (b) -----------------------------------------------------------------------
# (b) dokonać redukcji wymiarowości wszystkich obiektów w zbiorze do 2 najbardziej znaczących wymiarów,
#       za pomocą opracowanej funkcji wiPCA
# --------------------------------------------------------------------------------------------------------------------
X_pca, W, eig_vals = wiPCA(X, k=2)
# wywołanie funkcji PCA
#pca = PCA(n_components=2)
#X_pca = pca.fit_transform(X)

# Wykres redukcji wymiarowości obiektów na zbiore iris
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('2b: Wizualizacja obiektów z bazy iris')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()
# ---------------------------------------- (c) -----------------------------------------------------------------------
# (c) Zwizualizować elementy zbioru w przestrzeni cech z oznaczonymi klasami,
#       np. za pomcą kolorów, etykiet lub symboli (*, x, +, .)
# --------------------------------------------------------------------------------------------------------------------
# Wykres elementów zbioru w przestrzeni cech z oznaczonymi klasami
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
plt.title('2c: Elementy zbioru w przestrzeni cech z oznaczonymi klasami')
plt.xlabel('Długość')
plt.ylabel('Szerokość')
plt.colorbar()
plt.show()
# ====================================================================================================================

# ====================================================================================================================
# 3. Testowanie PCA na zbiorze digits
# ====================================================================================================================
# ---------------------------------------- (a) -----------------------------------------------------------------------
# (a) Wczytać zbiór digits (load digits)
digits = datasets.load_digits()
X = digits.data
y = digits.target
# ---------------------------------------- (b) -----------------------------------------------------------------------
# (b) Dokonać redukcji wymiarowości wszystkich obiektów w zbiorze do 2 najbardziej znaczących wymiarów
#     za pomocą opracowanej funkcji wiPCA
# --------------------------------------------------------------------------------------------------------------------
X_pca, W, eig_vals = wiPCA(X, k=2)
# wywołanie funkcji PCA
pca = PCA()
pca.fit(X)
#X_pca = pca.fit_transform(X)

# wykres redukcji wymiarowości obiektów na zbiorze digits
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('3b: Redukcja wymiarowości obiektów na zbiorze digits')
plt.xlabel('Pierwsza składowa główna')
plt.ylabel('Druga składowa główna')
plt.colorbar()
plt.show()
# ---------------------------------------- (c) -----------------------------------------------------------------------
# (c) Pokazać krzywą wariancji dla rosnącej liczby składowych głównych (tak jak na Rysunku 3)
# --------------------------------------------------------------------------------------------------------------------
# Wygenerowanie krzywej wariancji
variance = pca.explained_variance_ratio_
variance_cumulative = np.cumsum(variance)

# wykres krzywej wariancji
plt.plot(variance_cumulative)
plt.title('3c: Wizualizacja wariancji składowych głównych')
plt.xlabel('Numer składowej')
plt.ylabel('Skumulowana wariancja')
plt.show()
# ---------------------------------------- (d) -----------------------------------------------------------------------
# (d) Zwizualizować elementy zbioru w przestrzeni cech z oznaczonymi klasami (podobnie do Rysunku 4, funkcja scatter)
# --------------------------------------------------------------------------------------------------------------------
# Wykres punktowy przedstawiający rzut obiektów
plt.scatter(X_pca[:, 0], X_pca[:, 1],cmap='viridis', c=y )
plt.title('3d: wizualizacja obiektów z bazy digits')
plt.xlabel('Składowa 1')
plt.ylabel('Składowa 2')
plt.colorbar()
plt.show()
# ---------------------------------------- (e) -----------------------------------------------------------------------
# (e)  Wykonać eksperyment polegający na ocenie średniego błędu rekonstrukcji dla całego zbioru dla kolejno 
#      zwiększającej się liczby składowych głównych (można to zrobić za pomoca obliczania odległości dla wszystkich 
#      obiektów w bazie od ich zrekonstruowanych postaci - funkcja z Laboratorium nr 6) - przykładowy przebieg 
#      zmianności odległości na Rysunku 5. Zadanie to wymaga napisania funkcji obliczającej transformatę 
#      odwrotną do PCA, zwracającą obiekt(-y) o wymiarowości zgodnej z obiektem(-ami)#   
# --------------------------------------------------------------------------------------------------------------------
# Funkcja do obliczania transformaty odwrotnej do PCA
def inverse_transform_pca(pca, transformed):
    # pca         ==> Obiekt PCA dopasowany do zbioru danych.
    # transformed ==> Zbiór zredukowanych do n_components składowych głównych transformat z PCA.
    # Obliczenie transformaty do pierwotnej przestrzeni cech
    reconstructed = pca.inverse_transform(transformed)    
    # Zaokrąglenie do 4 miejsc po przecinku
    reconstructed = np.round(reconstructed, decimals=4) 
       
    # Funkcja zwraca zbiór zrekonstruowanych obiektów o wymiarowości zgodnej z pierwotnym zbiorem danych.
    return reconstructed
# --------------------------------------------------------------------------------------------------------------------
# Liczba składowych głównych
n_components = np.arange(1, 5)

# Obliczenie średniego błędu rekonstrukcji dla każdej liczby składowych głównych
mean_errors = []
for n in n_components:
    errors = []
    for i in range(10):  # powtarzamy 10 razy, aby uniknąć wyników losowych
        # PCA z n składowymi głównymi
        pca = PCA(n_components=n)
        # Dopasowanie PCA do zbioru iris
        pca.fit(iris.data)
        # Transformaty z PCA
        transformed = pca.transform(iris.data)
        # Transformaty z PCA zredukowane do pierwotnej przestrzeni cech
        reconstructed = inverse_transform_pca(pca, transformed)
        # Błąd rekonstrukcji
        error = np.mean(np.sum((iris.data - reconstructed) ** 2, axis=1))
        errors.append(error)
    mean_error = np.mean(errors)
    mean_errors.append(mean_error)

# Wykres zmienności średniego błędu rekonstrukcji w zależności od liczby składowych głównych
plt.plot(n_components, mean_errors, 'o-')
plt.title('3e: Wykres zmienności średniego błędu rekonstrukcji')
plt.xlabel('Liczba składowych')
plt.ylabel('Odleglość')
plt.show()
# ====================================================================================================================





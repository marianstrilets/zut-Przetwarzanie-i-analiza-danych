# ====================================================================================================================
#   Laboratorium 8 – algorytm k-środków
# Cel:
#   Zapoznanie się z algorytmem k–środków oraz jego implementacja.
# ====================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------- (1) -----------------------------------------------------------------------
#   1. Napisać funkcje:
# (a) d=distp(X,C,e), która wyliczy odległość euklidesową między dwomazbiorami punktów X i C
# (b) d=distm(X,C,V), która wyliczy odległość Mahalanobis’a między dwoma zbiorami punktów X i C;
#     V jest macierzą kowariancji zbioru X
# (c) [C,CX]=ksrodki(X,k), która dla zadanej macierzy wzorców X oraz liczby grup k, wyznaczy centra C i sąsiedztwa CX.
# --------------------------------------------------------------------------------------------------------------------

# Funkcja do obliczania odległości euklidesowej między dwoma zbiorami punktów X i C
def distp(X, C, e=0):
    d = np.zeros((len(C), len(X)))
    X = np.array(X)
    C = np.array(C)

    for k in range(len(C)):
        for i in range(len(X)):
            diff = X[i] - C[k]
            d[k][i] = np.sqrt(np.dot(diff, diff.T))
    return d

# Funkcja do obliczania odległości Mahalanobisa między dwoma zbiorami punktów X i C
def distm(X, C, V):
    d = np.zeros((len(C), len(X)))
    X = np.array(X)
    C = np.array(C)

    for k in range(len(C)):
        for i in range(len(X)):
            diff = X[i] - C[k]
            d[k][i] = np.sqrt(np.dot(diff, np.linalg.inv(V)))
            d[k][i] = np.sqrt(np.dot(d[k][i], diff.T))
    return d

# ====================================================================================================================
#   2. Zaimplementowac algorytm k–środków. W postaci zbioru X wybrać zbiór autos.
# ====================================================================================================================
# Funkcja do implementacji algorytmu k-środków
def ksrodki(X, k):
    centra = np.zeros([k, len(X[0])])
    for i in range(len(centra)):
        temp = X[np.random.randint(0, len(X))]
        while (temp == centra).any():
            temp = X[np.random.randint(0, len(X))]
        for j in range(len(centra[i])):
            centra[i][j] = temp[j]
    P = np.zeros((len(X), k))
    while True:
        P_temp = np.copy(P)
        C_temp = np.copy(centra)
        for i in range(len(X)):
            for j in range(k):
                dist = distp([X[i]], [centra[j]])
                dists = []
                for w in range(k):
                    dists.append(distp([X[i]], [centra[w]]))
                    if dist == min(dists):
                        P[i][j] = 1
                        for f in range(k):
                            if f != j:
                                P[i][j] = 0
                    else:
                        P[i][j] = 0
        for k_ in range(len(centra)):
            suma1 = 0
            suma2 = 0
            for i in range(len(X)):
                suma1 += P[i][k_] * X[i]
                suma2 += P[i][k_] 
        CX = [[] for i in range(k)]
        for kk in range(len(centra)):
            for i in range(len(P)):
                if P[i][kk]:
                    CX[kk].append(X[i])
        if np.array_equal(P, P_temp) and np.array_equal(centra, C_temp):
            break
    return centra, CX

# Wczytanie danych z pliku CSV
autos = pd.read_csv('autos.csv', na_values='', keep_default_na=False)
autos.fillna(0, inplace=True)

# Przygotowanie danych wejściowych
probki = np.zeros((len(autos['length']), 2))
for i in range(len(probki)):
    probki[i][0] = autos['wheel-base'][i]
    probki[i][1] = autos['length'][i]

# Wywołanie algorytmu k-środków
centra, CX = ksrodki(probki, 4)

# Przygotowanie danych do wykresu
points = CX  # Punkty przypisane do klastrów
centers = centra  # Centra klastrów

# Podział punktów na klastry
clusters = [[] for _ in range(len(centers))]
for i, cluster_points in enumerate(points):
    clusters[i] = np.array(cluster_points)

# Podział współrzędnych na osie x i y
x = [point[0] for point in centers]
y = [point[1] for point in centers]

#====================================================================================================================
#   3. Zilustrować graﬁcznie wyniki działania algorytmu.
#====================================================================================================================
# Wykres
plt.scatter(x, y, c='g', label='Cluster Centers')  # Centra klastrów
colors = ['b', 'r', 'y', 'purple']
for i, cluster in enumerate(clusters):
    x_cluster = [point[0] for point in cluster]
    y_cluster = [point[1] for point in cluster]
    plt.scatter(x_cluster, y_cluster, c=colors[i], label='Cluster {}'.format(i+1))  # Punkty w klastrach

plt.legend()
plt.xlabel('Wheel-Base')
plt.ylabel('Length')
plt.title('k-means Clustering')
plt.show()

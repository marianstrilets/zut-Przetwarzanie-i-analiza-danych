import numpy as np
import pandas as pd
# ------------------------------------------ 1 --------------------------------------------------------------------
# 1 . Używając standardowego słownika języka Python napisać funkcję [xi,ni]=freq(x, prob=True),
#   która dla zadanej kolumny danych x dyskretnych zwróci: unikalne wartości xi, ich estymowane
#   prawdopodobieństwa pi lub częstości ni.

def freq(x, prob=True):
    # Funkcja zwraca unikalne wartości kolumny x oraz ich estymowane prawdopodobieństwa pi lub częstości ni.
    # Jeśli prob=True, zwraca prawdopodobieństwa pi, a jeśli prob=False, zwraca częstości ni.
    xi, ni = np.unique(x, return_counts=True)
    if prob:
        pi = ni / np.sum(ni)
        return xi, pi  # zwraca unikalne wartości kolumny x, zwraca prawdopodobieństwa pi
    else:
        # zwraca unikalne wartości kolumny x, zwraca częstości ni.
        return xi, ni
# ================================================================================================================
# ------------------------------------------ 2 -------------------------------------------------------------------
# 2 . Napisać funkcję [xi, yi, ni] = freq2(x, y, prob=True), która dla zadanych kolumn danych x i y zwróci:
#   unikalne warto±ci atrybutów xi, yi oraz łączny rozkład częstości lub liczności ni (w zależności od prarametru prob).


def freq2(x, y, prob=True):
    #   Funkcja zwraca unikalne wartości atrybutów x i y oraz ich łączny rozkład częstości ni lub prawdopodobieństw pi.
    #   Jeśli prob=True, zwraca prawdopodobieństwa pi, a jeśli prob=False, zwraca częstości ni.
    # Tworzenie DataFrame z kolumnami x i y
    data = pd.DataFrame({'x': x, 'y': y})
    # Obliczanie łącznego rozkładu częstości lub prawdopodobieństw
    ni = data.groupby(['x', 'y']).size().reset_index(name='freq')
    # Rozdzielanie xi, yi i ni
    xi = ni['x'].tolist()
    yi = ni['y'].tolist()
    ni = ni['freq'].tolist()
    # Obliczanie prawdopodobieństw, jeśli prob=True
    if prob:
        pi = [f / sum(ni) for f in ni]
        # zwraca unikalne wartości atrybutów x i y oraz estymowane prawdopodobieństwo pi
        return xi, yi, pi
    else:
        # zwraca unikalne wartości atrybutów x i y oraz ich łączny rozkład częstości ni
        return xi, yi, ni
# ================================================================================================================
# ------------------------------------------ 3 -------------------------------------------------------------------
# 3. Wykorzystując powyższe funkcje, napisać funkcje, które wylicze: entropię h=entropy(x) oraz
#   przyrost informacji i=infogain(x, y) zgodnie ze wzorami


def entropy(x):
    # Entropia dla wektora x
    x = np.array(x)
    pi = x[x > 0] / np.sum(x)
    entropy = -np.sum(pi * np.log2(pi))
    return entropy


def infogain(x, y):
    # Przyrost informacji dla kolumn x i y
    _, Hx = freq(x)
    _, Hy = freq(y)
    _, _, Hxy = freq2(x, y)
    infogain = entropy(Hx) + entropy(Hy) - entropy(Hxy)
    return infogain


# ================================================================================================================
# ------------------------------------------ 4 -------------------------------------------------------------------
# 4. Wczytać dane testowe zoo.csv oraz dokonać selekcji/stopniowania atrybutów
#   z wykorzystaniem kryterium przyrostu informacji.
# Wczytanie danych i usunięcie pierwszej kolumny
data = pd.read_csv('zoo.csv', sep=",")
data = data.iloc[:, 1:]

table_ = np.array([])
entropy_ = np.array([])
infogain_ = np.array([])

# Dla każdej kolumny obliczenie entropii i przyrostu informacji
for feature in data.columns[:-1]:
    table_ = np.append(table_, feature)
    entropy_ = np.append(entropy_, entropy(
        freq(data[feature])[1]))         # obliczenie entropii
    # obliczenie przyrostu informacji
    infogain_ = np.append(infogain_, infogain(data[feature], data["type"]))

result = pd.DataFrame(
    {'Kolumna': table_, 'Entropia': entropy_, 'Przyrost informacji': infogain_})

zadanie4 = result
# ================================================================================================================
# ------------------------------------------ Wyswietlania wymików ------------------------------------------------
print('\n\t--------- Zadanie 1 ----------------------')
x = pd.array(np.random.randint(1, 20, size=100))
[xi, ni] = freq(x)                              #
print("\n", xi, "\n", ni)
print('\n\t--------- Zadanie 2 ----------------------')
x = pd.array(np.random.randint(1, 7, size=100))
y = pd.array(np.random.randint(1, 7, size=100))  #
[xi, yi, ni] = freq2(x, y)                       #
print("\n", xi,"\n", yi, "\n", ni)
print('\n\t--------- Zadanie 4 ----------------------')
print(zadanie4)

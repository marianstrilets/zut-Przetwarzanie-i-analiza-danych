import numpy as np

# ----------------------------- Tablice -------------------------------------------------
# Tablica jenowymiarowa
ta = np.array([1, 2, 3, 4, 5, 6, 7])

# Tablica dwuwymiarowa
tb = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Wykonaj transpozycję tablicy b za pomocą funkcji transpose
tb = np.transpose(tb)

# Utwórz i wyświetl tablicę składającą się ze 100 elementów za pomocą funkcji arange.
tc = np.arange(100)

# Utwórz i wyświetl tablicę składającą się z 10 liczb w zakresie od 0 do 2. Użyj funkcji linspace.
td = np.linspace(0, 2, 10)

# Za pomocą funkcji arange utwórz tablicę wartości od 0 do 100 ze skokiem wartości co 5.
te = np.arange(0, 100, 5)

# ----------------------------- Liczby losoqe -------------------------------------------
# Za pomocą funkcji random utwórz tablicę z 20 liczb losowych rozkładu normalnego, zaokrąglonych do dwu miejsc po przecinku.
l1 = np.round(np.random.normal(size=20), 2)

# Wygeneruj losowo 100 liczb całkowitych w zakresie od 1 do 1000.
l2 = np.random.randint(0, 1000, size=100)

# Za pomocą funkcji zeros i ones wygeneruj dwie macierze o rozmiarze 3 × 2.
m1 = np.zeros([3, 2])
m2 = np.ones([3, 2])

# Utwórz macierz losową złożoną z liczb całkowitych z zakresu od 0 do 100 o rozmiarze 5 × 5 i nadaj jej typ 32bit
mlos = np.random.randint(0, 100, size=(5, 5), dtype=np.int32)

# ----------------------------- Zadania -------------------------------------------------
# Wygeneruj tablicę złożoną z losowo wybranych liczb dziesiętnych z zakresu od 0 do 10 (tablica a).
za = np.random.uniform(0, 10, size=np.random.randint(0, 10))

# Zamień wartości na integer i wstaw w nową tablicę (tablicę b).
zb = za.astype(int)

# Znajdź funkcję numpy, która zaokrągli tablicę a do liczb całkowitych. Zamień je następnie na typ integer
za1 = np.round(za, 0)
zb1 = za1.astype(int)

# Porównaj wyniki z a i b i wyświetl wniosek za pomocą polecenia print.
print('------------------------------')
print('porownanie a i b do: ')
print(za)
print(za1)
print('porownanie a i b: ')
print(zb)
print(zb1)
print('------------------------------')
# ---------------------------------------------------------------------------------------
# ----------------------------- Selekcja danych -----------------------------------------
# wprowadź tablicę b=np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32).
# wprowadź tablicę b=np.array([[1,2,3,4,5], [6,7,8,9,10]],#dtype=np.int32).
sb = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)

# Za pomocą funkcji ndim sprawdź ile wymiarów ma tablica b.
sw = np.ndim(sb)

# Za pomocą funkcji size, sprawdź z ilu elementów składa się tablica b.
ss = np.size(sb)

# Wybierz wartości dwa i cztery z tablicy b.
sv = sb[:, [1, 3]]

# Wybierz pierwszy wiersz tablicy b.
swt = sb[0]

# Wybierz wszystkie wiersze z kolumny pierwszej tablicy b.
swk1 = sb[:, 0]

# Wygeneruj macierz losową o rozmiarze 20 × 7, złożoną liczb całkowi-tych w przedziale od 0 do 100.
# Wyświetl wszystkie wiersze dla czterech pierwszych kolumn
smacierz = np.random.randint(0, 100, size=(20, 7))
print(smacierz[:, np.arange(4)])
print('------------------------------')

# ---------------------------------------------------------------------------------------
# ----------------------------- Działania matematyczne i logiczne. ----------------------
# Stwórz dwie macierze liczb całkowitych z przedziału od 1 do 10 o rozmiarach 3 × 3 (a i b).
matrix1 = np.random.randint(1, 10, size=(3, 3))
matrix2 = np.random.randint(1, 10, size=(3, 3))

# Dodaj (za pomocą + oraz funkcii add), odejmij (za pomocą - oraz funkcii subtract),
# pomnóż (za pomocą * oraz funkcii multiply, dot, matmul), podziel (za pomocą / oraz funkcii divide),
# spotęguj (za pomocą ** oraz funkcii power) macierzy a i b przez siebie.
m_add = np.add(matrix1, matrix2)            # matrix1 + matrix2
m_subtract = np.subtract(matrix1, matrix2)  # matrix1 - matrix2
m_divide = np.divide(matrix1, matrix2)      # matrix1 / matrix2
m_pow = np.power(matrix1, matrix2)          # matrix1 ** matrix2
# --------------------------------------------------------------
m_multiply = np.multiply(matrix1, matrix2)  # matrix1 * matrix2
#m_dot = np.dot(matrix1, matrix2)
#m_matmul = np.matmul(matrix1, matrix2)
# np.multiply(a,b) każdy element macierzy pierwszej jest mnożony przez odpowiadający mu element macierzy drugiej.
# np.dot(a,b) wykonuje standardowe mnożenie macierzy, czyli mnoży pierwszą macierz przez transponowaną drugą macierz i sumuje iloczyny elementów.
# np.matmul(a,b) działa podobnie jak dot(), ale dla dwóch macierzy jednowymiarowych (wektorów) wykonuje ich mnożenie macierzowe, a dla macierzy wielowymiarowych wykonuje standardowe mnożenie macierzy.
# -------------------------------------------------------------

# Sprawdź czy wartości macierzy a są większe lub równe 4.
m_eq1 = matrix1 >= 4

# Sprawdź czy wartości macierzy a są większe bądź równe 1 i mniejsze bądź równe 4.
m_eq2 = np.logical_and(matrix1 >= 1, matrix1 <= 4)

# Znajdź funkcję w numpy do obliczenia sumy głównej przekątnej macierzy b.
m_diag = np.diag(matrix1) # diagonal
m_trace = np.trace(matrix1) # suma przekatnej

# ---------------------------------------------------------------------------------------
# ----------------------------- Dane statystyczne ---------------------------------------
# Oblicz sumę, wartość minimum, maksimum, odchylenie standardowe macierzy b.
x = np.random.randint(-9, 9, size=(4, 4))
suma = np.sum(x)
min = np.min(x)
max = np.max(x)
odchylenia_std = np.std(x)  # odchylenie standardowe macierzy
# ---------------------------------------------------------------------------------------
# ------------------  Rzutowanie wymiarów za pomocą rehape lub resize. ------------------
# Utwórz tablicę składającą się z 50 liczb (np., za pomocą funkcji arange).
tab = np.arange(50)

# Za pomocą funkcji reshape utwórz z tej tablicy macierz o wymiarach 10 × 5.
tab1 = tab.reshape([10, 5])

# Zrób to samo za pomocą funkcji resize.
tab2 = tab.resize(10, 5)

# Sprawdź do czego służy komenda ravel. Napisz wniosek.
# tablica wielowymiarowa zostaje przekształcona w jednowymiarową
tabr = np.ravel(tab)

# Stwórz dwie tablice o rozmiarach 5 i 4 (np., za pomocą funkcji arange) i dodaj je do siebie.
# W tym celu sprawdź do czego służy funkcja newaxis i wykorzystaj ją. Napisz wniosek.

# Funkcja newaxis jest bardzo przydatnym narzędziem w NumPy do modyfikowania wymiarów tablic.
# Dzięki niej można bardzo łatwo zmienić jednowymiarową tablicę na dwu- lub trójwymiarową,
# co często jest przydatne w obliczeniach numerycznych.
tb1 = np.arange(5)[:, np.newaxis]
tb2 = np.arange(4)[np.newaxis, :]
tb3 = tb1 + tb2
# ---------------------------------------------------------------------------------------
# --------------------------------- Sortowanie danych. ----------------------------------
# Sprawdź składnie funkcji sort i argsort.
# Wprowadź macierz a=np.random.randint(0, 100, size(5,5)).
s_tab = np.random.randint(0, 100, size=(5, 5))

# Funkcja sort w bibliotece NumPy służy do sortowania wartości w tablicy w kolejności rosnącej.
s_sort = np.sort(s_tab)
# Funkcja argsort zwraca indeksy, które sortują tablicę. Funkcja argsort zwraca tablicę indeksów, które muszą być użyte do posortowania tablicy wejściowej w kolejności rosnącej.
s_argsort = np.argsort(s_tab)

# Posortuj wiersze rosnąco.
sort_tab = np.sort(s_tab, axis=1, kind='mergesort')

# Posortuj kolumny malejąco.
sort_tab = np.sort(sort_tab, axis=0, kind="heapsort")[::-1]
# ---------------------------------------------------------------------------------------

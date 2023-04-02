import numpy as np

print("=============================== 1 ==============================")
# 1). Utwórz macierz składającą się z pięciu kolumn i 10 wierszy losowo wy-
# branych liczb całkowitych z zakresu od 0 do 100 i policz sumę głównej
# przekątnej tej macierzy, używając funkcji trace a następnie wyświetl
# wartości używając funkcji diag.
z1_matrix = np.random.randint(0, 101, size=(10, 5))
z1_trace = np.trace(z1_matrix)
z1_diag = np.diag(z1_matrix)

print("Macierz losowych liczb całkowitych:")
print(z1_matrix)
print("Suma głównej przekątnej macierzy: ", z1_trace)
print("Wartości na głównej przekątnej macierzy: ", z1_diag)

print("=============================== 2 ==============================")
# 2). Utwórz dwie tablice wymiaru 5 × 5 z losowo wybranych liczb dziesiętnych 
# z rozkładu normalnego i przemnóż je przez siebie.
z2_table1 = np.random.normal(size=(5, 5))
z2_table2 = np.random.normal(size=(5, 5))
z2_suma = z2_table1 * z2_table1

print("Tablica a:")
print(z2_table1)
print("Tablica b:")
print(z2_table2)
print("Tablica wynikowa c:")
print(z2_suma)

print("=============================== 3 ==============================")
# 3). Utwórz dwie tablice z losowo wybranych liczb całkowitych w zakresie
# od 1 do 100. Stwórz z nich macierze o 5 kolumnach i dodaj te macierze
# do siebie.
z3_tab1 = np.random.randint(1, 101, size=(1, 5))
z3_tab2 = np.random.randint(1, 101, size=(1, 5))

print("Tablica a:\n", z3_tab1)
print("Tablica b:\n", z3_tab2)

z3_matrix1 = np.reshape( z3_tab1, (5, -1))
z3_matrix2 = np.reshape( z3_tab2, (5, -1))
z3_result = z3_matrix1 + z3_matrix2

print("Macierz wynikowa:\n", z3_result)

print("=============================== 4 ==============================")
# 4).  Stwórz dwie macierzy: jedną o 5 kolumnach i 4 wierszach oraz drugą o
# 4 kolumnach i 5 wierszach. Dodaj je do siebie używając transformacji
# wymiarów za pomocą jednego ze znanych sposobów
matrix_A = np.random.randint(1, 101, size=(4, 5))
matrix_B = np.random.randint(1, 101, size=(5, 4))

result = matrix_A + np.transpose(matrix_B)

print(matrix_A)
print(matrix_B)
print(result)

print("=============================== 5 ==============================")
# 5).  Pomnóż kolumny 3 i 4, stworzonych przez siebie macierzy.
product = result[:, 2] * result[:, 3]
print("Wynik mnożenia kolumn 3 i 4 macierzy A i B:")
print(product)

print("=============================== 6 ==============================")
# 6). Wygeneruj dwie macierze o rozkładzie normalnym (np.random.normal)
# i jednostajnym(np.random.uniform).
# Policz wartości średnie, odchylenie standardowe, wariancje, sumy, war-
# tości minimalne i maksymalne. Wyniki wyświetl.
normal_matrix1 = np.random.normal(0, 1, (3, 3))
normal_matrix2 = np.random.normal(2, 3, (3, 3))
uniform_matrix1 = np.random.uniform(0, 1, (3, 3))
uniform_matrix2 = np.random.uniform(2, 3, (3, 3))

print("Macierz o rozkładzie normalnym 1:\n", normal_matrix1)
print("Średnia: ", np.mean(normal_matrix1))
print("Odchylenie standardowe: ", np.std(normal_matrix1))
print("Wariancja: ", np.var(normal_matrix1))
print("Suma: ", np.sum(normal_matrix1))
print("Wartość minimalna: ", np.min(normal_matrix1))
print("Wartość maksymalna: ", np.max(normal_matrix1))

print("\nMacierz o rozkładzie normalnym 2:\n", normal_matrix2)
print("Średnia: ", np.mean(normal_matrix2))
print("Odchylenie standardowe: ", np.std(normal_matrix2))
print("Wariancja: ", np.var(normal_matrix2))
print("Suma: ", np.sum(normal_matrix2))
print("Wartość minimalna: ", np.min(normal_matrix2))
print("Wartość maksymalna: ", np.max(normal_matrix2))

print("\nMacierz o rozkładzie jednostajnym 1:\n", uniform_matrix1)
print("Średnia: ", np.mean(uniform_matrix1))
print("Odchylenie standardowe: ", np.std(uniform_matrix1))
print("Wariancja: ", np.var(uniform_matrix1))
print("Suma: ", np.sum(uniform_matrix1))
print("Wartość minimalna: ", np.min(uniform_matrix1))
print("Wartość maksymalna: ", np.max(uniform_matrix1))

print("\nMacierz o rozkładzie jednostajnym 2:\n", uniform_matrix2)
print("Średnia: ", np.mean(uniform_matrix2))
print("Odchylenie standardowe: ", np.std(uniform_matrix2))
print("Wariancja: ", np.var(uniform_matrix2))
print("Suma: ", np.sum(uniform_matrix2))
print("Wartość minimalna: ", np.min(uniform_matrix2))
print("Wartość maksymalna: ", np.max(uniform_matrix2))

print("=============================== 7 ==============================")
# 7). Wygeneruj dwie macierze kwadratowe a i b (o wymiarach zdecyduj
# się samodzielnie), pomnóż je przez siebie używając (a*b) oraz funkcji
# dot. zobacz Jaka jest różnica? Napisz kiedy warto wykorzystać funkcję dot?
matrixA = np.random.rand(3, 3)
matrixB = np.random.rand(3, 3)
matrixC1 = matrixA * matrixB
matrixC2 = np.dot(matrixA, matrixB) 

print("Wynik mnożenia macierzy a i b za pomocą operatora mnożenia:")
print(matrixC1)
print("Wynik mnożenia macierzy a i b za pomocą funkcji dot:")
print(matrixC2)
print("Różnica między operatorem mnożenia a funkcją dot polega na tym, że każdy element macierzy a przez odpowiadający mu element macierzy b zwraca macierz o takim samym wymiarze jak macierze wejściowe, podczas gdy funkcja dot wykonuje mnożenie macierzy zgodnie z zasadami algebry liniowej i zwraca macierz wynikową o wymiarze, który jest wynikiem operacji na wymiarach macierzy wejściowych.")
print("Funkcja dot jest szczególnie przydatna w przypadku mnożenia macierzy, gdy chcemy obliczyć iloczyn skalarny wektorów, transponować macierz lub wykonać bardziej skomplikowane operacje algebry liniowej\n")


print("=============================== 8 ==============================")
# 8). Sprawdź funkcję strides oraz as strided. Zastosuj je do wyboru
# danych z macierzy np. 5 kolumn z trzech pierwszych wierszy.
matrix = np.random.randint(0, 101, size=(10,10))
rows, cols = matrix.shape
num_cols = 5
row_stride, col_stride = matrix.strides
new_matrix = np.lib.stride_tricks.as_strided(matrix,
                                             shape=(rows, num_cols),
                                             strides=(row_stride, col_stride),
                                             writeable=False)[:3, :]
print("Wynik macierz do wyboru danych z macierzy:\n", matrix)
print("Wynik macierz po wyboru danych z macierzy:\n", new_matrix)


print("=============================== 9 ==============================")
# 9). Wygeneruj dwie tablice a i b.
# Połącz je z użyciem funkcji vstack i stack. Czym one się różnią?
# Zastanów się i napisz, w jakich przypadkach warto je zastosować?
tabA = np.random.randint(0, 101, size=(5,5))
tabB = np.random.randint(0, 101, size=(5,5))
tabC = np.vstack((tabA, tabB))
tabC1 = np.stack((tabA, tabB), axis=0)


print("tabA:\n", tabA)
print("tabB:\n", tabB)
print("tabC:\n", tabC)
print("tabC1:\n", tabC1)

print('''Różnica między tymi funkcjami polega na tym, że vstack dokonuje połączenia wzdłuż pierwszego wymiaru, 
a stack pozwala na wybór dowolnego wymiaru. Dlatego też funkcja vstack jest szybsza i bardziej przejrzysta, 
jeśli chodzi o połączenie tablic wzdłuż jednego wymiaru. Warto zastosować funkcję vstack, 
gdy chcemy połączyć dwie tablice o takiej samej liczbie kolumn, ale różnych wierszach. 
Przykładem może być połączenie tabeli danych zawierającej wyniki testów dla różnych osób z tabelą zawierającą ich imiona.
Natomiast funkcja stack jest bardziej ogólna i można jej użyć do łączenia tablic w różny sposób. 
Przykładem może być łączenie dwóch tablic, w których jedna ma dwa wymiary, a druga jeden wymiar, 
lub łączenie tablic o różnych kształtach poprzez dodanie nowego wymiaru.''')


print("=============================== 10 ==============================")
# 10).  Użyj funkcji strides oraz as strided do obliczenia wartości maksymalnej bloków danych z macierzy (zob. rysunek)
matrix = np.arange(24).reshape(4, 6)
block_size = (3, 2)
row_stride, col_stride = matrix.strides
block_row_stride, block_col_stride = row_stride * block_size[1], col_stride * block_size[0]
blocks = np.lib.stride_tricks.as_strided(matrix, shape=(2, 2, 3, 2), strides=(block_col_stride, col_stride, block_row_stride, row_stride))

max_values = np.max(blocks, axis=(2, 3))

for i in range(2):
    for j in range(2):
        print(f"max({blocks[i, j].flatten()}) = {max_values[i, j]}")

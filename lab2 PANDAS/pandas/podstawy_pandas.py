import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# --------------------------------------  Manipulowanie danymi. ------------------------
# Zapoznaj się z obiektem pandas.DataFrame i utwórz tabelę złożoną z liczb losowych rozkładu
# normalnego złożoną z trzech kolumn z nagłówkiem (A, B, C) i pięciu wierszy,
# z indeksem o nazwie data złożonym z dat w przedziale od 2020-03-01 do 2020-03-05,
# pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)

# utworzenie tabeli złożonej z liczb losowych z rozkładu normalnego np.random.normal(0, 1, (5, 3))
# pd.DataFrame(data = None, index=None, columns=None, dtype=None, copy=None)
data = np.random.normal(0, 1, size=(5, 3))
df = pd.DataFrame(data)
df.columns = ['A', 'B', 'C']
df.index = pd.date_range('2020-03-01', '2020-03-05')
df.index.name = 'data'
print(df)
del(df, data)
print('--------------------------------------------------------')
# Wygeneruj tabelę złożoną z liczb losowych i indeksie id w formacie integer złożoną
# z 20 wierszy i trzech kolumn (’A’,’B’,’C’). Następnie:
data = np.random.randint(-99, 99, size=(20, 3))
df = pd.DataFrame(data)
df.columns = ['A', 'B', 'C']
print(df, end='\n\n')

print('--------------------------------------------------------')
# wybierz trzy pierwsze wiersze z tabeli, za pomocą metody head()
df.head(3)

#  wybierz trzy ostatnie wiersze z tabeli,
df.iloc[-3:]

# wyświetl nazwę indeksu tabeli,
df.index.name = 'id'
df.index.name

# wyświetl nazwy kolumn
df.columns

# wyświetl tylko dane tabeli bez indeksów i nagłówków kolumn
df.to_string(index=False, header=False)

# wybierz pięć losowo wybranych wierszy
df.sample(n=5)

# wybierz tylko wartości kolumny ’A’ a następnie tylko ’A’ i ’B’ (skorzystaj z values),
df['A'].values
df[['A', 'B']].values

# Zapoznaj się z funkcją iloc i wyświetl: trzy pierwsze wiersze i kolumny ’A’ i ’B’,
df[['A', 'B']].head(3)

# Zapoznaj się z funkcją iloc i wyświetl: wiersze 0,5,6,7 i kolumny 1 i 2
tmp = df.iloc[[0, 5, 6, 7], [1, 2]]
print(tmp)
print('--------------------------------------------------------')

#  Zapoznaj się z funkcją describe i wyświetl podstawowe statystyki tabeli:
print(df.describe())
#  sprawdź które dane w tabeli są większe od 0
df[df > 0]

#  wyświetl tylko dane większe od 0,
df[df > 0].values

# Wybór wartości większych od 0 z kolumny 'A'
df[df['A'] > 0]['A']

# Średnia w kolumnach
df.mean()

# Średnia w wierszach
df.mean(axis=1)
del(df, tmp)

# Zapoznaj się z funkcją concat. Utwórz dwie dowolne tabele (o wymiarach zdecyduj się samodzielnie)
# i połącz je ze sobą. Dokonaj transpozycji nowej tabeli.
df1 = pd.DataFrame(data=np.random.randint(
    0, 10, size=(5, 3)), columns=list('ABC'))
df2 = pd.DataFrame(data=np.random.randint(
    10, 20, size=(5, 2)), columns=list('DE'))

df = pd.concat([df1, df2], axis=1)

print(df1)
print(df2)
print(df)
del(df1, df2, df)
print('--------------------------------------------------------')
# --------------------------------------------------------------------------------------
# --------------------------------------  Sortowanie -----------------------------------
# W tabelach DataFrame mogą być umieszczone różne typy danych:
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [
                  'a', 'b', 'a', 'b', 'b']}, index=np.arange(5))
df.index.name = 'id'
print(df)

# Aby posortować dane w tabeli DataFrame po kolumnie lub indeksie, można skorzystać z funkcji sort_values().
# –posortuj dane po ’id’ malejąco,
df.sort_index(ascending=False)

# –posortuj dane po kolumnie ’y’ rosnąco.
df.sort_values('y')
del(df)
# --------------------------------------------------------------------------------------
# --------------------------------------  Grupowanie danych. ---------------------------
slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'],
           'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'],
           'Pound': [10, 15, 50, 40, 5],
           'Profit': [20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day', 'Fruit']).sum())
del(slownik, df3)
# --------------------------------------------------------------------------------------
# --------------------------------------   Wypełnianie danych. -------------------------
df = pd.DataFrame(np.random.randn(20, 3),
                  index=np.arange(20), columns=list('ABC'))
df.index.name = 'id'

# Wykonaj i opisz za pomocą funkcji print jak działają poniższe komendy:
df['B'] = 1
df.iloc[1,2] = 10
df[df<0] =-df
del(df)
# --------------------------------------------------------------------------------------

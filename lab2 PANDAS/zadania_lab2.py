import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

data = {'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']}
df = pd.DataFrame(data=data)

# ------------------------------------ 1 ----------------------------------------------------------
# 1. Zgrupować tabele po zmiennej symbolicznej Y, a następnie wyznaczyć średnią wartość 
# atrybutu numerycznego X w grupach wyznaczonych przez Y.
grouped = df.groupby('y')
result = grouped.mean()
print(result)

# ------------------------------------ 2 ----------------------------------------------------------
# 2. Wyznaczyć rozkład liczności atrybutów (value_counts).
value_counts = df['y'].value_counts()
print(value_counts)

# ------------------------------------ 3 ----------------------------------------------------------
# 3. Wyczytać dane autos.csv, za pomocą polecenia np.loadtxt oraz pandas.read csv. Sprawdź różnice.
filename = 'autos.csv'
data_pd = pd.read_csv(filename)
print(data_pd)

# ------------------------------------ 4 ----------------------------------------------------------
# 4. Zgrupować ramkę danych po zmiennej ’make’ a następnie wyznaczyć średnie 
# zużycie paliwa dla każdego z producentów.
res1  = data_pd.groupby('make')['city-mpg'].mean()
print(res1)

# ------------------------------------ 5 ----------------------------------------------------------
# 5. Zgrupować ramkę danych po zmiennej ’make’ liczności dla atrybutu ’fuel-type’
res2 = data_pd.groupby('make')['fuel-type'].value_counts()
print(res2)
# ------------------------------------ 6 ----------------------------------------------------------
# 6. Dopasować wielomian 1 i 2 stopnia prognozujący wartość zmiennej ’city-mpg’, 
# względem ’length’ (np.polyfit , np.polyval)
x = np.linspace(data_pd['length'].min(), data_pd['length'].max(), 100)

fit1 = np.polyfit(data_pd['length'], data_pd['city-mpg'], 1)
p1 = np.polyval(fit1, x)
fit2 = np.polyfit(data_pd['length'], data_pd['city-mpg'], 2)
p2 = np.polyval(fit2, x)

# ------------------------------------ 7 ----------------------------------------------------------
# 7. Wyznaczenie współczynnika korelacji liniowej pomiędzy tymi zmiennymi (scipy.stats)
corel_pearsona = stats.pearsonr(data_pd['length'], data_pd['city-mpg'])[0]
print('Współczynnik korelacji liniowej:', corel_pearsona)

# ------------------------------------ 8 ----------------------------------------------------------
# 8. Zwizualizować wyniki dopasowania, zaznaczając próbki oraz dopasowane 
# krzywe na tle próbek dla zmiennych ’city-mpg’, ’length’.

fig, ax = plt.subplots()
# Wyświetlenie dopasowanych krzywych
ax.scatter(data_pd['length'], data_pd['city-mpg'], s=10, label='Dane')
ax.plot(x, p1, 'r', label='Wielomian 1. stopnia')
ax.plot(x, p2, 'g', label='Wielomian 2. stopnia')

# Dodanie legendy i tytułu osi
ax.legend()
ax.set_xlabel('Length')
ax.set_ylabel('City MPG')
ax.set_title('Wyniki dopasowania')

plt.show()
# ------------------------------------ 9 ----------------------------------------------------------
# 9. Dla zmiennej ’length’ utworzyć jednowymiarowy estymator funkcji gęstości, 
# w tym celu użyć scipy.stats.gaussian kde.
# Zwizualizować wynik przedstawiając jednocześnie próbki i funkcję gęstości. 
# Do wykresu dodać legendę. W tym celu użyć (plot(...,label=’...’), legend)

# Utworzenie jednowymiarowego estymatora funkcji gęstości dla zmiennej "length"
kde = gaussian_kde(data_pd['length'])

# Wyznaczenie wartości na podstawie estymatora funkcji gęstości
x = np.linspace(data_pd['length'].min(), data_pd['length'].max(), 100)
y = kde(x)

# utworzenie dwóch wykresów na jednym oknie graficznym
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Wykres przedstawiający próbki oraz funkcję gęstości
ax1.plot(x, y, label='Gęstość')
ax1.hist(data_pd['length'], density=True, alpha=0.3, label='Próbki')
ax1.legend()
ax1.set_xlabel('Długość')
ax1.set_ylabel('Gęstość')
ax1.set_title('Estymatora funkcji gęstości')

# ------------------------------------ 10 ---------------------------------------------------------
# Wyznaczenie wartości na podstawie estymatora funkcji gęstości
kde = gaussian_kde(data_pd['width'])
x = np.linspace(data_pd['width'].min(), data_pd['width'].max(), 100)
y = kde(x)
ax2.plot(x, y, label='Gęstość')
ax2.hist(data_pd['width'], density=True, alpha=0.3, label='Próbki')
ax2.legend()
ax2.set_xlabel('Szerokość')
ax2.set_ylabel('Gęstość')
ax2.set_title('Estymatora funkcji gęstości')

# Wyświetlenie wykresów
plt.show()
# ------------------------------------ 11 ---------------------------------------------------------
# 11. Utworzyć dwuwymiarowy estymator funkcji gęstości dla zmiennych
# ’width’ i ’length’, wynik przedstawić graﬁcznie w nowym
# oknie rysując próbki poleceniem plot oraz funkcję gęstości używając polecenia meshgrid i contour.
# Wynik zapisać do plików w formacie *.png i *.pdf (savefig).

# Utworzenie dwuwymiarowego estymatora funkcji gęstości
kde = gaussian_kde([data_pd['width'], data_pd['length']])

# Wyznaczenie wartości na podstawie estymatora funkcji gęstości
x, y = np.meshgrid(np.linspace(data_pd['width'].min(), data_pd['width'].max(), 100),
                   np.linspace(data_pd['length'].min(), data_pd['length'].max(), 100))
z = kde([x.ravel(), y.ravel()]).reshape(x.shape)

# Utworzenie nowego okna rysunkowego
fig, ax = plt.subplots()

# Wykres próbek
ax.plot(data_pd['width'], data_pd['length'], 'k.', markersize=2)

# Wykres funkcji gęstości
ax.contour(x, y, z, levels=10, cmap='inferno')

# Dodanie tytułu i etykiet osi
ax.set_title('Estymator dwuwymiarowej funkcji gęstości')
ax.set_xlabel('Width')
ax.set_ylabel('Length')

# Zapis wykresu do plików PNG i PDF
plt.savefig('estymator_dwuwymiarowy.png', dpi=300, format='png')
plt.savefig('estymator_dwuwymiarowy.pdf', format='pdf')

# Wyświetlenie wykresu
plt.show()


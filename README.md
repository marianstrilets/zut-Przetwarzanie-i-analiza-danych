##### Laboratorium 1: Podstawy zastosowania NUMPY

Moduł "NUMPY: jest zestawem narzędzi umożliwiającym zaawansowane obliczenia matematyczne. Rozszerza on możliwości PYTHON o nowe typy danych, operacje na nich oraz funkcje przyśpieszające obliczenia.

##### Laboratorium 2: PANDAS

Moduł "Pandas" jest zestawem narzędzi umożliwiającym sprawne manipulowanie zestawem danych. Rozszerza on możliwości PYTHON o łatwy import i eksport danych m.in. do plików w formatach tekstowych (csv) czy konkretnych aplikacji (excel). Dane są przechowywane w tabelach tzw. DataFrame. "Pandas" dostarcza wiele narzędzi do selekcji, łączenia i sortowania danych.

##### Laboratorium 3-4: Dyskretyzacja, kwantyzacja, binaryzacja

Celem laboratorium jest poznanie zagadnień związanych z konwersją analogowo-cyfrową, oraz problemów które może ona dostarczyć. Dodatkowo poruszony jest temat binaryzacji, jako "uproszczonej kwantyzacji" do dwóch wartości [0, 1] na rzeczywistym przykładzie segmentacji obiektu z tła na obrazie.

##### Laboratorium 5: Selekcja zmiennych za pomoc¡ przyrostu informacji

Celem ćwiczenia jest analiza jednej z podstawowych metod selekcji zmiennych dyskretnych opartej na przyroście informacji. W zadaniu wykorzystywane będą pakiety numpy i scipy.sparse. Do wczytywania danych, można posłużyć się pakietami pandas (metoda read_csv) oraz sklearn.

##### Laboratorium 6-7: Analiza głównych składowych PCA

Celem ćwiczenia jest implementacja algorytmu Analizy Głównych Składowych PCA dla danych wielowymiarowych. Należy wykonać własną implementację w formie funkcji wiPCA, która przyjmie na wejściu zbiór danych w formie macierzy oraz liczby wymiarów docelowych. Na wyjściu powinna znalezć się macierz zerdukowanej przestrzeni cech, macierz wektorów własnych, wektor liczb własnych oraz dodatkowo średni wektor wejściowy. Napisana własnoręcznie funkcję porównać z implementacją dostępną w sklearn, naprzykład:

```python
from sklearn import datasets
from sklearn.decomposition import PCA
iris = datasets.load iris()
pca = PCA(n components=2)
X r = pca.fit(X).transform(X)
```

##### Laboratorium 8
##### Laboratorium 9
##### Laboratorium 10
##### Laboratorium 11

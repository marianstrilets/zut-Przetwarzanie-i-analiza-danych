#### Przetwarzanie i analiza danych

##### Struktura plików:

```bash
├── lab-1: Podstawy zastosowania NUMPY
├── lab-2: PANDAS
├── lab-3: 
├── lab-4: Dyskretyzacja, kwantyzacja, binaryzacja
├── lab-5: Selekcja zmiennych za pomocą przyrostu informacji
├── lab-6: 
├── lab-7: Analiza głównych składowych PCA
├── lab-8: 
├── lab-9: Algorytm K-środków
├── lab-10: 
├── lab-11: Algorytm K-NN
├── lab-12: 
├── lab-13: Fltracja przebiegów czasowych(audio), zastosowanie okien kroczących, FFT
```

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

##### Laboratorium 8-9:

Zapoznanie się z algorytmem k–środków oraz jego implementacja

##### Laboratorium 10-11:

Celem laboratorium jest zapoznanie się z algorytmem najbliższych sąsiadów.
Zaimplementowanie metody k-nn i użycie jej w zadaniu klasyﬁkacji oraz
regresji. Przystosowanie algorytmu do korzystania z kD-drzew.

##### Laboratorium 12-13: Fltracja przebiegów czasowych(audio), zastosowanie okien kroczących, FFT 

Celem laboratoriów jest zapoznanie się z możliwościami analizy i ekstrakcji informacji z przebiegów czasowych, na przykładzie sygnału audio. Wszystkie ćwiczenia powinny być wykonywane na własnym nagraniu (patrz punkt 1.1), a w razie problemów z wydzieleniem odpowiednich fragmentów posiłkować się będzie można dodatkowymi nagraniami zawierającymi wybrane głoski nagrane indywidualnie.

##### Laboratorium 14:

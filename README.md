# Inteligentny System Zarządzania Temperaturą

## Języki

- [English](README_EN.md)
- [Polski](README.md)

## Opis

Projekt ma na celu stworzenie modelu AI do systemu zarządzania temperaturą w budynku z wykorzystaniem technik uczenia maszynowego, a konkretnie modelu A3C (Asynchronous Advantage Actor-Critic). System ma za zadanie automatycznie regulować ogrzewanie, aby utrzymać optymalną temperaturę określoną przez użytkownika, jednocześnie minimalizując częstość przełączania ogrzewania i odchylenie od zadanej temperatury. Projekt umożliwia symulowanie wymiany ciepła między budynkiem a otoczeniem, ale wierne odzwierciedlenie fizyki tego procesu NIE jest (na razie) przedmiotem tego projektu. 

Dla celów porównawczych powstał prosty model dwustanowy, który włącza ogrzewanie gdy temperatura spada poniżej ustalonego poziomu (histereza 1°C). Porównanie tych modeli w przygotowanym środowisku możliwe jest dzięki wygenerowanym danym dostępnym w lokalizacji: ```data/plots/``` oraz ```data/table/```

### Co udało się osiągnąć

Wyszkolony model przełącza ogrzewanie pomiędzy stanem włączonym / wyłączonym (np. zawór dwu-stanowy). Model umożliwia analizę sytuacji (stanu) dla **dowolnej** liczby pomieszczeń o zbliżonej charkterystyce i określeniu najlepszej akcji (włącz / wyłącz). Model operuje na danych zebranych z 7 godzin w 10-minutowych odstępach (42 wektory). Dane te są traktowane jako aktualny stan zgodnie z procesem decyzyjnym Markova. (To podejście ma niewielki skutek uboczny: przez pierwsze 7 godzin model operuje na niepełnych danych, jednak nie powoduje to znaczącego obniżenia jakości predykcji)

### Co można zrobić

- Dopracować symulator, żeby lepiej odzwierciedlał rzeczywistość (wymiana ciepła, otwarte okna, nagłe zmiany pogody)
- Spróbować zmniejszyć model, lub zwiększyć jego dokładność.
- Przeszkolić model, który będzie regulował ogrzewanie w sposób płynny (procent otwarcia: 0-100%)

## Komponenty

Projekt składa się z kilku kluczowych komponentów:

### Moduł Symulacji Środowiska

Moduł ten, oparty na równaniach różniczkowych, symuluje dynamikę temperatury w budynku. Dane o temperaturach i stanie ogrzewania są przekazywane do środowiska pracy modelu, symulatora lub cichego trybu. Ten ostatni zapisuje dane do pliku oraz generuje wykres.

### Równania różniczkowe modelu

Równanie opisujące zmianę temperatury w pomieszczeniu w czasie jest zdefiniowane jako:

$$
\frac{dT}{dt} = \frac{1}{C} \left(\eta \cdot (H(t) - T(t)) - k \cdot A \cdot (T(t) - T_{\text{zew}}(t))\right) 
$$

#### Legenda:

- \( T(t) \): Temperatura wewnętrzna pomieszczenia w czasie \( t \) [°C].
- \( H(t) \): Temperatura podłogi (czyli źródła ciepła) w czasie \( t \) [°C].
- \( T_{\text{zew}}(t) \): Temperatura zewnętrzna w czasie \( t \) [°C].
- \( \eta \): Współczynnik efektywności przenikania ciepła z podłogi do powietrza w pokoju [W/m²K].
- \( k \): Współczynnik przenikania ciepła przez ściany budynku [W/m²K].
- \( A \): Całkowita powierzchnia ścian zewnętrznych pomieszczenia [m²].
- \( C \): Pojemność cieplna pomieszczenia, określająca ilość energii potrzebnej do podgrzania całego powietrza w pomieszczeniu o jeden stopień Celsjusza [J/K].

#### Wyjaśnienie równania:

Równanie to opisuje, jak szybko temperatura w pomieszczeniu zmienia się w odpowiedzi na działanie systemu ogrzewania podłogowego oraz wymianę ciepła z otoczeniem zewnętrznym. Współczynnik \( \eta \) mierzy, jak efektywnie ciepło jest przekazywane z podłogi do powietrza w pomieszczeniu, a współczynnik \( k \) odzwierciedla, jak szybko ciepło ucieka z pomieszczenia przez ściany zewnętrzne. Pojemność cieplna \( C \) mówi o tym, jak duża jest zdolność pomieszczenia do magazynowania ciepła.


Równanie opisujące zmianę temperatury podłogi H(t) w czasie:

$$
\frac{dH}{dt} = \alpha (H_{\text{max}} - H(t)) - \beta (H(t) - T(t))
$$

gdzie:
- \(H_{\text{max}}\) jest stałą reprezentującą maksymalną temperaturę, jaką może osiągnąć podłoga,
- \(\alpha\) to współczynnik szybkości ogrzewania podłogi,
- \(\beta\) to współczynnik szybkości chłodzenia podłogi,
- \(T(t)\) to temperatura otoczenia wewnątrz budynku.



### Analiza Danych i Wizualizacja

Dla trybu cichego wykorzystano biblioteki Pandas i Matplotlib. Dane są analizowane i wizualizowane w formie wykresów (jedno pomieszczenie), co pozwala na ocenę efektywności systemu ogrzewania oraz strategii zarządzania temperaturą.

### Interaktywna Wizualizacja w Pygame

Dodatkowy moduł napisany w Pygame zapewnia interaktywną wizualizację pracy modelu w czasie rzeczywistym, z możliwością przyspieszenia upływu czasu. Umożliwia to obserwowanie efektów działania systemu w przystępnej formie wizualnej. (Czerwona kropka w pokoju oznacza grzanie, szara brak ogrzewania)

## Technologie

- Python
- TensorFlow
- Numpy
- Pandas
- Matplotlib
- Pygame

## Uruchomienie Projektu

### Uruchomienie symulacji
```python run_simulator.py```

### Uruchomienie cichego trybu
```python run_silent_mode.py```

### Uruchomienie treningu modelu A3C
```python run_training.py```

## Licencja

- [Licencja](LICENSE)


---

Projekt jest połączeniem uczenia maszynowego, fizyki i wizualizacji danych, mającym na celu przygotowanie modelu do efektywnego zarządzanie temperaturą.


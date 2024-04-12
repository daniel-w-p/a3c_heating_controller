# Inteligentny System Zarządzania Temperaturą

## Języki

- [English](README_EN.md)
- [Polski](README.md)

## Opis

Projekt ma na celu stworzenie inteligentnego systemu zarządzania temperaturą w budynku z wykorzystaniem technik uczenia maszynowego, a konkretnie modelu A3C (Asynchronous Advantage Actor-Critic). System ma za zadanie automatycznie regulować ogrzewanie, aby utrzymać optymalną temperaturę określoną przez użytkownika, jednocześnie minimalizując częstość przełączania ogrzewania.

## Komponenty

Projekt składa się z kilku kluczowych komponentów:

### Moduł Symulacji Środowiska

Moduł ten, oparty na równaniach różniczkowych, symuluje dynamikę temperatury w budynku. Dane o temperaturach i stanie ogrzewania są zapisywane do tabeli z pomocą Pandas, co umożliwia ich dalszą analizę.

### Równania różniczkowe modelu

Równanie opisujące zmianę temperatury wewnętrznej T(t) w czasie:

$$
\frac{dT}{dt} = h(t) - k \cdot (T(t) - T_{\text{out}}(t))
$$

gdzie:
- \(h(t)\) reprezentuje wpływ ogrzewania (może być funkcją zależną od temperatury podłogi \(H(t)\)),
- \(k\) to współczynnik przenikania ciepła przez ściany budynku,
- \(T_{\text{out}}(t)\) to temperatura zewnętrzna.

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

Wykorzystując bibliotekę Matplotlib, dane są analizowane i wizualizowane w formie wykresów, co pozwala na ocenę efektywności systemu ogrzewania oraz strategii zarządzania temperaturą.

### Interaktywna Wizualizacja w Pygame

Dodatkowy moduł napisany w Pygame zapewnia interaktywną wizualizację pracy modelu w czasie rzeczywistym, z możliwością przyspieszenia upływu czasu. Umożliwia to obserwowanie efektów działania systemu w przystępnej formie wizualnej.

## Technologie

- Python
- TensorFlow
- Pandas
- Matplotlib
- Pygame

## Uruchomienie Projektu

### Uruchomienie symulacji
```python run_simulator.py```

### Uruchomienie treningu modelu A3C
```python run_training.py```

## Licencja

- [Licencja](LICENSE)


---

Projekt jest połączeniem uczenia maszynowego, fizyki i wizualizacji danych, mającym na celu poprawę efektywności energetycznej budynków poprzez inteligentne zarządzanie temperaturą.


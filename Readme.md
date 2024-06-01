# Narzędzie do Półautomatycznej Adnotacji Tekstu

## Autorzy

### EnjoyersIT:

* Rafał Budnik
* Zuzanna Damszel
* Ireneusz Okniński

---

## TL;DR

Aplikacja umożliwia adnotowanie danych tekstowych w celu uczenia modeli NLP.
Użytkownik może wczytać zbiór danych, zdefiniować adnotacje, iteracyjnie adnotować
tekst. W celu usprawnienia procesu ręcznej adnotacji, użytkownik może douczać model na podstawie
zaakceptowanych adnotacji. Użytkownik ma możliwość korzystania z przygotowanego modelu BiLSTM,
który będzie uczył się od zera na podstawie zaakceptowanych adnotacji, możliwe jest wczytanie stanu modelu z pliku.
Aplikacja umożliwia również zdefiniowanie własnego klasyfikatora oraz podobnie jak w przypadku
modelu BiLSTM, możliwe jest załadowanie jego stanu z pliku. Proces uczenia może byc na bieżąco monitorowany
poprzez wyświetlanie statystyk dotyczących modelu i procesu uczenia.

---

## Instalacja

Aplikacja jest budowana przy użyciu narzędzia `make`.
Wspieranym systemem operacyjnym jest `Linux`.
Aplikacja jest kompatybilna z `Python 3.12`, pozostałe wersje nie są gwarantowane.

### Instalacja zależności

`make build`

### Uruchomienie aplikacji

`make app`

### Uruchomienie testów

`make test`

### Uruchomienie testów z pokryciem

`make test-cov`

### Uruchomienie lintera

`make lint`

### Uruchomienie formatera

`make format`

### Utworzenie dokumentacji

`make docs`

---

## Funkcjonalności

1. **Tworzenie projektu**
    - Użytkownik może stworzyć nowy projekt, w którym będzie mógł przeprowadzić proces adnotacji.
    - Wymagane jest podanie:
        * Nazwy
        * Opisu
        * Modelu
            - Jeżeli wybrano własny model, należy podać ścieżkę do pliku z modelem. (`.py`)
        * Wybór lokalizacji zapisu projektu
        * Format pliku wyjściowego (`.csv`, `.json`)
    - Opcjonalnie:
        * Wczytanie stanu modelu z pliku (`.pth`)

2. **Użycie własnego modelu**
    - Użytkownik może skorzystać z własnego modelu, który zostanie użyty do adnotacji tekstu
    - Wymagane jest dostarczenie pliku z modelem w formacie `.py`, w którym zdefiniowana jest dokładnie jedna
      klasa
      dziedzicząca po
      `torch.nn.Module` i implementująca metody `forward`.
    - Model musi posiadać:
        * Wejście: `Embedding(100_000, ...)`
        * Wyjście: `Linear(..., liczba_kategorii)`
            - `liczba_kategorii` = 2 * liczba_adnotacji + 1 (każda adnotacja ma przypisaną kategorię `B` i `I`,
              dodatkowo jest kategoria `<O>`)
    - Przykładowy plik z modelem:
      ```python
      import torch.nn as nn
            
      class YourModule(nn.Module):
          def __init__(
              self
          ):
              super(YourModule, self).__init__()
              self.embedding = nn.Embedding(100_000, ...)
              # Define your layers here
              self.linear = nn.Linear(..., 5)
      
          def forward(self, x):
              # Define forward pass here
              return self.linear(x)
      ```
    - **UWAGA:** Obecnie nie jest możliwe przekazanie parametrów do modelu. Wymagane jest ich zdefiniowanie w kodzie
      modelu.

3. **Wczytanie wag do modelu**
    - Użytkownik może wczytać stan modelu z pliku w formacie `.pth`
    - Wczytane wagi zostaną załadowane do modelu i będą użyte w procesie adnotacji
    - Wagi muszą być zgodne z modelem, do którego są wczytywane
    - Sposób zapisu wag:
   ```python
   model_state = {
     "model_state_dict": model.state_dict(),
     "optimizer_name": type(optimizer).__name__,
     "optimizer_state_dict": optimizer.state_dict(),
     "loss_name": type(loss).__name__,
     "loss_state_dict": loss.state_dict()
   }
   torch.save(model_state, 'model.pth')
   ```
    - Wymagane jest podanie `model_state_dict`, pozostałe klucze są opcjonalne.
    - Zalecany sposób użycia tej funkcjonalności:
        1. Utworzenie modelu bez wczytywania wag
        2. Pobranie `model.pth` oraz `label_to_idx.json` z folderu projektu
        3. Wytrenowanie modelu, zmiana parametrów stanu itp.
        4. Utworzenie nowego projektu i zaimportowanie wcześniej zapisanych wag.
            - Możliwa jest też podniana pliku `model.pth` w folderze projektu.

4. **Wczytywanie zbioru danych:**
    - Obsługiwane formaty: `.csv`, `.json`
        - `.csv`: Przykładowy plik CSV może wyglądać tak:
            ```
            Zaawansowane	programowanie	w	Pythonie	.
            Named	entity	recognition
            ```
          Każdy fragment tekstu jednorazowo wyświetlany użytkownikowi jest zapisany w jednej linii pliku CSV.
          Wyrazy muszą być oddzielone znakiem tabulacji (`\t`).
        - `.json`: Przykładowy plik JSON może wyglądać tak:
            ```json
            [
                ["Zaawansowane", "programowanie", "w", "Pythonie", "."],
                ["Named", "entity", "recognition"]
            ]
            ```
      Plik .json ma postać listy list, gdzie każda lista zawiera pojedynczy fragment tekstu.
    - Możliwość rozbudowy o inne formaty w przyszłości

5. **Definiowanie adnotacji:**
    - Użytkownik może wprowadzić listę adnotacji, które mają być oznaczone w tekście
    - Możliwe jest zdefiniowanie maksymalnie 20 adnotacji.
    - Użytkownik może zdefiniować nazwę adnotacji oraz kolor, w jakim ma być oznaczona
    - Przykładowe adnotacje:
        - `PER` - imię i nazwisko
        - `LOC` - nazwa lokalizacji
        - `ORG` - nazwa organizacji
        - `MISC` - inne

6. **Wczytanie istniejącego projektu**
    - Użytkownik może wczytać istniejący projekt, aby kontynuować proces adnotacji.
    - Wymagane jest podanie ścieżki do folderu z projektem.
    - Wczytane dane zostaną załadowane do aplikacji, a użytkownik będzie mógł kontynuować proces adnotacji w miejscu,
      w którym został przerwany.
    - Stan modelu zostanie załadowany z pliku.

7. **Adnotacja tekstu**
    - Użytkownikowi wyświetlany jest fragment tekstu, który musi zostać oznaczony zgodnie ze zdefiniowanymi adnotacjami.
    - W górnej części ekranu wyświetlane są adnotacje, które użytkownik może przypisać do wyrazów.
    - W dolnej części ekranu wyświetlany jest fragment tekstu, który użytkownik może adnotować.
    - Przyciski
        - `AI Assistant` - włącza lub wyłącza podpowiedzi od modelu AI
        - `Stats` - przejście do statystyk projektu
        - `Save` - zapisuje obecny stan projektu
        - `Accept` - zapisuje adnotacje i przechodzi do kolejnego fragmentu tekstu
        - `Reset` - resetuje adnotacje w obecnym fragmencie tekstu
        - `Multiword Mode` - włącza lub wyłącza tryb adnotacji wielowyrazowej
    - Instrukcja adnotacji:
        1. Kliknij na etykietę, którą chcesz przypisać do wyrazu.
        2. Jeżeli `Multiword Mode` jest wyłączone:
            - kliknięcie `Lewym przyciskiem` przypisze etykietę do jednego wyrazu
            - kliknięcie `Prawym przyciskiem` usunie etykietę z wyrazu
        3. Jeżeli `Multiword Mode` jest włączone:
            - kliknięcie `Lewym przyciskiem` rozpocznie adnotację wielowyrazową
            - kliknięcie `Prawym przyciskiem` zakończy adnotację wielowyrazową
            - maksymalnie można przypisać etykietę do 5 wyrazów
    - Przy próbie wyjścia z aplikacji użytkownik zostanie zapytany, czy chce zapisać obecny stan projektu.

8. **Zapis adnotacji**
    - Adnotacje są zapisywane w formacie `.csv` lub `.json`
    - Przykładowy plik `.csv`:
      ```
      Dr	Jan	Kowalski	pracuje	w	Google	.	B-PER	B-PER	I-PER	<O>	<O>	B-ORG	<O>
      ```
    - Przykładowy plik `.json`:
      ```json
      [
          ["Dr", "Jan", "Kowalski", "pracuje", "w", "Google", ".", "B-PER", "B-PER", "I-PER", "<O>", "<O>", "B-ORG", "<O>"]
      ]
      ```
    - Adnotacje są zapisywane w formacie: `[wyraz1, wyraz2, ..., etykieta1, etykieta2, ...]`
    - Aby wydobyć adnotacje z pliku, należy podzielić listę na pół, gdzie pierwsza połowa to wyrazy, a druga to
      etykiety.
    - W przypadku adnotacji wielowyrazowej, etykieta `B-...` jest przypisywana do pierwszego wyrazu, a etykieta `I-...`
      do kolejnych wyrazów.
    - W przypadku braku adnotacji, przypisywana jest etykieta `<O>`

9. **Statystyki:**
    - Użytkownik może sprawdzić statystyki dotyczące projektu
    - Wyświetlane statystyki:
        - Liczba adnotacji dla każdej z kategorii
        - Liczba zaadnotowanych fragmentów tekstu
        - Liczba pozostałych(niezaadnotowanych) fragmentów tekstu
        - Metryki modelu(obliczone na podstawie zapisanych adnotacji):
            - Accuracy
            - Precision
            - Recall
            - F1

---

## Demo

Zapoznaj się z prezentacją aplikacji na filmie poniżej:

[![Demo](https://i9.ytimg.com/vi_webp/QO-ccS6-LK8/mq3.webp?sqp=CPSq7LIG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYACwAWKAgwIABABGBUgcigRMA8=&rs=AOn4CLA56h4h97ShBDzqyp8Fw3kbDNo80Q)](https://youtu.be/QO-ccS6-LK8)
---

## Możliwe kierunki rozwoju

- Usprawnienie dodawania własnych modeli, możliwość ich personalizacji
- Model w formie API, dzięki czemu użytkownik mógłby korzystać z modelu bez konieczności jego wczytywania
- Dodanie możliwości podglądu i edycji adnotacji
- Ulepszenie strategii wyboru fragmentów tekstu do adnotacji
- Dodanie większej liczby statystyk
- Dodanie większej liczby modeli do wyboru

---

## Wkład

Jeżeli chcesz pomóc w rozwoju projektu, skontaktuj się przedstawicielem zespołu.

`01158211@pw.edu.pl` - Ireneusz Okniński

---

## Licencja

Projekt jest dostępny na licencji MIT.

---

## Podziękowania

Serdeczne podziękowania dla naszych opiekunów, którzy pomogli nam w realizacji projektu.

* dr inż. Łukasz Neumann
* dr inż. Mateusz Modrzejewski

---
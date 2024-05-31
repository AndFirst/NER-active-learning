# Narzędzie do półautomatycznej adnotacji tekstu

## Opis
NER Active Learning to narzędzie do półautomatycznej adnotacji tekstu, skoncentrowane na rozpoznawaniu nazwanych encji (NER, ang. Named Entity Recognition). Projekt ten ma na celu wspomaganie procesu adnotacji tekstów poprzez wykorzystanie technik uczenia aktywnego, co pozwala na bardziej efektywne i szybkie etykietowanie dużych zbiorów danych tekstowych.

## Zbiór danych
Narzędzie przyjmuje zbiór danych w formacie .csv oraz .json.

### Przykład csv
```csv
Thousands	of	demonstrators	have	marched	through	London	to	protest	the	war	in	Iraq	and	demand	the	withdrawal	of	British	troops	from	that	country	.
Iranian	officials	say	they	expect	to	get	access	to	sealed	sensitive	parts	of	the	plant	Wednesday	,	after	an	IAEA	surveillance	system	begins	functioning	.
```
### Przykład json
```json
[
    [
        "Thousands",
        "of",
        "demonstrators",
        "have",
        "marched",
        "through",
        "London",
        "to",
        "protest",
        "the",
        "war",
        "in",
        "Iraq",
        "and",
        "demand",
        "the",
        "withdrawal",
        "of",
        "British",
        "troops",
        "from",
        "that",
        "country",
        "."
    ],
    [
        "Iranian",
        "officials",
        "say",
        "they",
        "expect",
        "to",
        "get",
        "access",
        "to",
        "sealed",
        "sensitive",
        "parts",
        "of",
        "the",
        "plant",
        "Wednesday",
        ",",
        "after",
        "an",
        "IAEA",
        "surveillance",
        "system",
        "begins",
        "functioning",
        "."
    ]
]
```
## Działanie
1. Użytkownik ładuje zbiór danych do modelu.
2. Zbiór zostaje przetworzony do postaci zrozumiałej dla modelu.
3. Model wybiera N zdań:   
    a. losowo,
    b. takich, które najsłabiej rozumie.
4. Model przekazuje użytkownikowi zdania do adnotacji.
5. Użytkownik adnotuje zdania, a następnie są one zwracane modelowi.
6. Zdania zostają zapisane w annotaded_data.
7. Model uczy się na batchu z powyżej wymienionych zdań.
8. Powtarzane są kroki od 3. do 7., dopóki nie wyczerpie się cały zbiór danych.

## Wkład
Jeśli chcesz wnieść swój wkład do projektu, prosimy o otwarcie zgłoszenia (issue) lub przesłanie prośby o połączenie (pull request) z odpowiednimi zmianami.

## Licencja
Projekt jest dostępny na licencji MIT. Szczegóły znajdują się w pliku LICENSE.
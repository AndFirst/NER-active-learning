# Narzędzie do półautomatycznej adnotacji tekstu

## Opis
Z powodu kryzysu nie ma opisu.

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
1. Użytkownik ładuje swój zbiór danych do modelu
2. Zbiór zostaje przetworzony do postaci zrozumiałej dla modelu.
3. Model wybiera N zdań:   
    a. losowo
    b. takich które najsłabiej rozumie
4. Model wysyła userowi zdania do adnotacji
5. User adnotuje zdania i odsyła userowi.
6. Zdania zostają zapisane w annotaded_data
7. Model uczy się na batchu z tych zdań.
8. Krok 3. dopóki nie wyczerpie się cały zbiór danych
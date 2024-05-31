# ZPRP-NER-Active-Learning
==============================

## Project description
NER Active Learning is a tool for semi-automated text annotation, focusing on Named Entity Recognition (NER). This project aims to facilitate the annotation process by leveraging active learning techniques, making it more efficient and faster to label large text datasets.

## Project organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Dataset
The tool accepts datasets in .csv and .json formats.

### CSV example
```csv
Thousands	of	demonstrators	have	marched	through	London	to	protest	the	war	in	Iraq	and	demand	the	withdrawal	of	British	troops	from	that	country	.
Iranian	officials	say	they	expect	to	get	access	to	sealed	sensitive	parts	of	the	plant	Wednesday	,	after	an	IAEA	surveillance	system	begins	functioning	.
```

### JSON example
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
## How does it work
1. The user uploads the dataset to the model.
2. The dataset is processed into a format understandable by the model.
3. The model selects N sentences:
    a. randomly,
    b. those it understands the least.
4. The model provides the sentences to the user for annotation.
5. The user annotates the sentences, which are then returned to the model.
6. The sentences are saved in annotated_data.
7. The model trains on a batch of the annotated sentences.
8. Steps 3 to 7 are repeated until the entire dataset is exhausted.

## Contributing
If you would like to contribute to the project, please open an issue or submit a pull request with your changes.

## License
This project is licensed under the MIT License. Details can be found in the LICENSE file.
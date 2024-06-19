Explainable Image Aesthetics Assessment
==============================

This is the repository for Master Thesis: Explainble Image Aesthetics Assessment.

This work explores the feasibility of an Explainable and Personalized Image Aesthetics Assessment (IAA). First, two Image Quality Assessment systems were implemented: a CNN-based model NIMA and a Transformer-based model MusiQ, with MusiQ outperforming NIMA on a comparative distortion evaluation. The Personalized framework was then designed and implemented as a modified Siamese neural network, using two image inputs and the NIMA model as an image encoder, inspired by how photographers compare two images at a time. The model was then personalized on four tasks, increasing in complexity: Learning to discriminate Black-and-White images, Blurry images, Blob-Overlaid images, and to prefer smiling over frowning faces. The personalization proved successful, with higher accuracy rates of all four personalized versions. The Explainable framework was implemented using LIME with default and AI-generated perturbations, first tested to explain a simple brightness-prediction model, then applied onto the two base models and onto the personalized versions. The interface of the Explainable Personalized IAA system was implemented into the Capture One photo-editing desktop application, where it was well-received by participants.

Project Organization
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

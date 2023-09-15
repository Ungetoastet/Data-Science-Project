Data Science Project
==============================

This is a template for your final data science project. It offers a structure for your project, but does not have to be followed strictly. In case your project has different needs, feel free to adapt it.

Data Sources
------------

A list of potential datasets to use can be found [here](https://github.com/awesomedata/awesome-public-datasets). You can also suggest your own dataset, but it needs to be approved.

Before you start working on your project, your research question and hypotheses need to approved first.

Requirements
------------

The README.md needs to contain the instructions to download all data and reproduce your experiments. All analyses and results need to be reproducible and documented in notebooks. In a sense, the notebooks should act as a final report. They should contain your hypotheses, methods, experiments, visualizations, analyses, plus results and also explain these. They should be understandable to a third party who has not worked on the project.

Your data science project must contain the following elements:

- A clearly stated problem definition as well as at least three hypotheses that you want to test.
- At least one model that you have trained and evaluated.
- At least one statistical significance test.
- At least one visualization.

Grading
------------

To pass, your project must be fully reproducible without errors. We recommend cloning your project, running it in a fresh environment, and following the instructions you provide in this README before submitting to ensure reproducibilty.

Otherwise, your project will be graded based on:

1. The quality of the documentation and code.
   - Are all steps documented?
   - Is the code understandable? (Add comments and references where necessary)
2. Understandibility of the experiments and results. For example:
   - Why did you opt for a particular method?
   - What does a particular visualization show?
   - What do your results mean?

In particular, we will **not** be grading based on the results of your experiments. You can get full points, as long as you can explain why you did what you did. We will not penalize you if your results are not statistically significant or you applied an incorrect method or test. On the other hand, applying correct methods will certainly not hurt your grade 😉

Project Organization
------------

    ├── data
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── src                <- Source code for use in this project. These are just example files
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    ├── README.md          <- Top-level README explaining the project and how to run the code.
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`

Reproduciblity Instructions
------------

To be filled out by the student.

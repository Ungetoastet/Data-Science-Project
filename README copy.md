# Data Science Project

This is a template for your final data science project. It offers a structure for your project but does not have to be followed strictly. In case your project has different needs, feel free to adapt it.

## Data Sources

Here's a list of several good places to look for datasets:

 - [awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)
 - [kaggle](https://www.kaggle.com/datasets)
 - [huggingface](https://huggingface.co/datasets)

## Research Question and Hypotheses

Before you start working on your project, you must formulate research questions and connected hypotheses. These need to be approved, before you fully start working.

What makes a good research question or hypothesis and what is the difference between the two?

A research question is a general question that describes what topic or aspect should be explored or investigated. For example: "What are the characteristic differences between different iris flower types?" or "Is there a relationship between flower color and petal length?". Hypotheses are more concrete and make assumptions about the relationship between variables or phenomena. For example: "Iris flower A has longer petals than iris flower B" or "Yellow flowers have longer petals than blue flowers". Importantly, hypotheses are empirically/experimentally verifiable! Whereas a research question guides the direction of your project, hypotheses guide the methods and experiments you will conduct.

## Example

Here is an example project description with research questions and hypotheses:

```md
In my project I will investigate if it is possible to predict the genre of a song based on its lyrics. As a basis I will use the [Spotify Tracks](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) dataset and scrape the correpsonding lyrics from [Genius](https://genius.com/). I will then train a classifier to predict the genre of a song based on its lyrics. I will use the following research questions and hypotheses to guide my project:

Research Questions:
   - RQ1: Are there prototypical words/topics for different genres?
   - RQ2: Is it possible to predict the genre of a song based on its lyrics?
   - RQ3: How well does an unsupervised clustering of songs based on their lyrics match the genre labels?

Hypotheses:
   - H1: A songs word frequency distribution is a better predictor for a song's genre than the the number of words in a song.
   - H2: A supervised classifier can predict the genre of a song better than an unsupervised clustering.
```

## Requirements

The README.md needs to contain the instructions to download all data and reproduce your experiments. All analyses and results need to be reproducible and documented in notebooks. In a sense, the notebooks should act as a final report. They should contain your hypotheses, methods, experiments, visualizations, analyses, plus results and also explain these. They should be understandable to a third party who has not worked on the project.

Your data science project must contain the following elements:

- A clearly stated problem definition as well as at least three hypotheses that you want to test.
- At least one model that you have trained and evaluated.
- At least one statistical significance test.
- At least one visualization.

## Grading

To pass, your project must be fully reproducible without errors. We recommend cloning your project, running it in a fresh environment, and following the instructions you provide in this README before submitting it to ensure reproducibility.

Your project will be graded based on the quality of the documentation of the experiments and results. Use the following questions to guide your project:

- Data:
  - What are the main properties of your data? 
  - What features does it have?
- Questions / Hypotheses:
  - What are your research questions and hypotheses and how did you test them?
  - Why did you opt for a particular method?
- Results:
  - What does a particular metric, visualization, or hypothesis test show?
  - What do your results suggest in the context of your questions / hypotheses?

In particular, we will **not** be grading based on the results of your experiments. You can get full points, as long as you can explain why you did what you did. We will not penalize you if your results are not statistically significant or if you applied an incorrect method or test. On the other hand, applying correct methods will certainly not hurt your grade 😉

## Project Organization

The following is a suggested project structure. If you have a different structure in mind, feel free to adapt it.

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

## Reproduciblity Instructions

To be filled out by the student.

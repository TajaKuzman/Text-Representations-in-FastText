# Text-Representations-in-FastText
Analysis of feature importance for genre identification through data transformation

 ## Task Description

In this task, I will analyse what importance different linguistic features have for the task of the automatic (web) genre identification (AGI) by comparing the performance of machine learning models, trained on various text representations. With this approach, I will be able to discover to which extent are lexical, grammatical and other features important for the identification of genre.

I will perform text classification with the linear model fastText. For the experiments, I will use the Slovene Web genre identification corpus GINCO 1.0  which consists of 1002 texts, manually annotated with 24 genre labels.
I will train and test the model on:
* baseline: plain text as extracted from the web during the creation of web corpora (used in previous experiments)
* lower-cased
* reduced to lemmas
* transformed into part-of-speech tags
* with function words (stopwords) removed
* consisting only of the words belonging to a certain word type, i.e. only nouns, only verbs, only adjectives, etc.

The setups will be compared based on micro and macro F1 scores, to measure the models’ performance on the instance level and the label level, and confusion matrices.

## Steps

### Data Preparation, Experiment Setup

Data:
* deduplicated GINCO corpus --> 983 texts
* original stratified train-dev-test split (60:20:20)

Steps:
* Transforming the GINCO JSON file to a FastText format
* Optimising FastText - hyperparameter search on dev split
* Preliminary experiments with data (on test split) to optimize the FastText performance (the performance of the baseline needs to be high enough so that the differences of performance on transformations, which we expect to be lower, will be visible):
    * Removing noise: 1) label Other 2) instances with secondary labels, 3) instances with secondary labels + tertiary labels, 4) instances with secondary label + tertiary labels + hard parameter
    * Downcasting: 1) Removing infrequent classes - using top 5-10 classes 2) Merging classes

#### Experiment Setup Conclusions
* At least 5 training runs for each experiment

### Experiments on Text Representations
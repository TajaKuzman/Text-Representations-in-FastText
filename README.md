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
See the notebook *1-Preparing_Data_Hyperparameter_Search.ipynb* where I found the best hyperparameters for the task, and *2-Language-Processing-of-GINCO.ipynb* where I linguistically preprocessed data with the CLASSLA pipeline.

Data:
* GINCO corpus with "keep" texts (reasons: more text, but duplicates omitted as they can be unrepresentative for the genre type)
* smaller number of labels: downsampled 12 set, labels with too few instances discarded, fuzzy labels (*Other*, *List of Summaries/Excerpts*) discarded, texts marked with *Hard* discarded --> 5 labels, 688 texts
* original stratified train-dev-test split (60:20:20): 410:141:137

Preliminary experiments:
* Optimising FastText - hyperparameter search on dev split --> average micro and macro F1 scores of 0.625 +/- 0.0036 and 0.618 +/- 0.003

#### Experiment Setup Conclusions
* Experiments on no. of epochs --> 350 epochs used
<img style="width:80%" src="experimental-setup-results\Number-of-epochs.png">
* Experiments on learning rate --> lr = 0.7
<img style="width:80%" src="experimental-setup-results\Learning-rate.png">
* Experiments on number of word n-grams used --> suprisingly, using unigrams (default) gives the best results
<img style="width:80%" src="experimental-setup-results\Ngrams.png">
* Default context window (5)

### Experiments on Text Representations

**Results**:
* baseline text: micro F1: 0.56 +/- 0.0, macro F1: 0.589 +/- 0.0
* lower-cased:  micro F1: 0.553 +/- 0.0045, macro F1: 0.587 +/- 0.009 - slightly lower results
* punctuation removed: micro F1: micro F1: 0.58 +/- 0.0028, macro F1: 0.616 +/- 0.0024 - improved results, especially forum (see the graph)
* numbers removed: micro F1: 0.583 +/- 0.0028, macro F1: 0.595 +/- 0.0025 - slight improvement, except in Forum, where it is worse
* lower-cased, punctuation removed, numbers removed: micro F1: 0.56 +/- 0.0, macro F1: 0.598 +/- 0.0 - no improvements in micro level, very slight improvements in macro level - improvement in forum, otherwise mostly no

* lemmas: micro F1: 0.597 +/- 0.0053, macro F1: 0.601 +/- 0.0035 - significant improvement over the baseline, especially for Information/Explanation, Promotion, for News no change, for Forum and Opinion worse
* part-of-speech tags (upos): micro F1: 0.54 +/- 0.0053, macro F1: 0.547 +/- 0.0056 - decrease overall, but in News and Opinion increase
* morphosyntactic descriptors (MSD): micro F1: 0.563 +/- 0.0072, macro F1: 0.536 +/- 0.019, increase in micro, decrease in macro, improvement in News and Information, high variation in Forum
* syntactic dependencies: micro F1: 0.61 +/- 0.0, macro F1: 0.639 +/- 0.00044 - the best results, high improvement, especially in News, Forum, Opinion. Decrease in Promotion.

Reduced features (lemmas in selected PoS class used, other replaced with O):
* only open class words - stopwords removed (ADP, AUX, CCONJ and SCONJ, DET, NUM, PART and PRON)
* only stop words
* only classes which denote subjectivity - ADJ, ADV, PART
* only PROPN, NOUN and VERB
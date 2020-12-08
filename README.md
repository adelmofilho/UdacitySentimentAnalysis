# Sentiment Analysis Deployment Project
### *Udacity Machine Learning Nanodegree*

<br>

## Before we start...

The notebook `Sagemaker Project.ipynb` and the codes on `train` and `server` folder were completed (questions and todo codes) withouy any improvement in order to keep a baseline to be compared with proposed modifications on notebook `Sagemaker Project Enhanced.ipynb` and codes on `enhanced` folder.

## What makes this project special?

- Website apparence was improved by the use of [boostrap library](https://getbootstrap.com/)

- User experience was improved by the use of the raw model response instead of binary response, which allows a continuous interpreation of results meaning.

- Modelling and data preparation methodologies were improved by:

    - Addition of a validation set to prevent overfitting;
    - Implementation of hyperparameter tunning jobs to search for the best model hyperparameters;
    - The most frequent and less-informative words are removed from the vocabulary

## Description
*TL;DR*: blog.adelmofilho.com/UdacitySentimentAnalysis



<center>
<img src="https://i.imgur.com/89xoYJv.png">
</center>

## Answered Questions

### Above we mentioned that `review_to_words` method removes html formatting and allows us to tokenize the words found in a review, for example, converting entertained and entertaining into entertain so that they are treated as though they are the same word. What else, if anything, does this method do to the input?  

Despite Html removing and word tokenizing, `review_to_words` method removes stopwords (i.e. very common words from a language) to reduce noise from data, lowercases all words in order to avoid distinct tokeninzation of a same word and, finnaly, splits each words as a list element.

### What are the five most frequently appearing (tokenized) words in the training set? Does it makes sense that these words appear frequently in the training set?

The table below displays the five most frequently tokenized words appearing in the training set.

| word | count |
|------|-------|
| movi | 51695 |
| film | 48190 |
| one  | 27741 |
| like | 22799 |
| time | 16191 |

The tokens `movi` (from movie), `film` (from film) and `one` are highly expected at any reviews as they are commonly used to reference the movie people are writing reviews.

The token `like` makes direct reference to the sentiment of the person writing reviews and the token `time` has multiple meanings on a reviews: movie total time, in expressions like `save your time`, some momment at the filme etc.

### In the cells above we use the `preprocess_data` and `convert_and_pad_data` methods to process both the training and testing set. Why or why not might this be a problem?

Despite to process both the training and testing set, the word dictionary (input of both methods) came only from the training set. In other words, both methods apply a transform operation based on the training set only. It imples on no problems as it creates no data leakage for the models.

### How does this model compare to the XGBoost model you created earlier? Why might these two models perform differently on this dataset? Which do you think is better for sentiment analysis?

Both algorithms were implemented as classification models were predictions lies between 0 and 1, and data preparation methodology was the same for both model training. The main differences come on the archicteture and optimization behind these algorithms.

XGBoost models are ensembles of weak tree based models that ensure low variance as primary objective and seek to reduce bias increasing the total number of estimators (trees). 

LSTM models are recurrent neural networks where data flows across multiple activation functions differently from classical neural networks where the signal across multiples neuron layers is vanished, on LSTM vanish gradient problems are solved which allows multiple hidden layers and therefore to incorporate more complexity.

Because the way each model deals with its internal parameter optimization it's expected different performances dispite the same dataset.

Finnaly, "which model is better?" is relative. But, as this is a sentimental analysis problem and interpretation is something desirable, one would pick a XGBoost model because its simplicty over a LSTM model.
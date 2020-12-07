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


## 

## Answered Questions

### Above we mentioned that `review_to_words` method removes html formatting and allows us to tokenize the words found in a review, for example, converting entertained and entertaining into entertain so that they are treated as though they are the same word. What else, if anything, does this method do to the input?  




### What are the five most frequently appearing (tokenized) words in the training set? Does it makes sense that these words appear frequently in the training set?

### In the cells above we use the `preprocess_data` and `convert_and_pad_data` methods to process both the training and testing set. Why or why not might this be a problem?

### How does this model compare to the XGBoost model you created earlier? Why might these two models perform differently on this dataset? Which do you think is better for sentiment analysis?
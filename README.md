# Deep Learning with Python
Workbook for the book Deep Learning with Python by Francois Chollet (`ISBN 978-1-61729-443-3`)

### Donations

I write these tutorials in my spare time because I want to make it easier for others to get started in the wonderful world of software engineering. If you've enjoyed my videos and found my code useful, please consider sending me a small donation using the link below :)

<https://www.paypal.me/reverendandrewmills>

## 01 - IMDB Movie Review Classification

### Binary Classification Problem

This script uses the IMDB dataset from Keras to train and test a movie review classification model. The script creates graphs to show loss and accuracy over epochs. Finally, we run the model against test data and print the decoded reviews, along with a score between 0 (negative review) and 1 (positive review). Reviews with a score of .25 or under are considered "negative" and reviews with a score of .75 or over are considered "positive".

## 02 - Reuters News Wire Classification

### Single-Label Multiclass Classification Problem

This script uses the Reuters dataset from Keras to train and test a newswire classification model. The script creates graphs to show loss and accuracy over epochs. Finally, we run the model against test data and print the decoded articles, along with the most likely topic that the article belongs to (out of a set of 46 topics). Reference topics were obtained from the [Reuters Analysis Repo](https://github.com/SteffenBauer/KerasTools/tree/master/Reuters_Analysis).

## 03 - Predicting House Prices with Regression

### Continuous Data Prediction Problem

This script uses the Boston House Prices dataset to train and test a house price regression model. If no flags are used when calling the script, it will use k-fold validation to split the dataset into 4 partitions and run 100 epochs of each set to calculate the Mean Absolute Error and Mean Squared Error. If the flag `predict` is added at the end of the script call, the script will skip this section and train over the whole training dataset with 100 epochs and print predictions for the first 10 house prices.

## 04 - Reading handwritten digits from the MNIST Dataset

### Single-Label Multiclass Classification Problem

This is a basic example of a Convolutional Neural Network (CNN). The script trains and tests against the MNIST dataset, before running predictions on the test dataset. The first 25 results are output to a `.png` image file.
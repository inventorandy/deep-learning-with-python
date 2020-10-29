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
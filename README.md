# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The data is a bank marketing dataset [from UCI](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), which contains information on Portuguese bank marketing to try and sell bank term deposits.  The data contains a target column, 'y', which is yes or no, and other columns with demographics and information on the custometrs.  We are trying to predict the 'y' column (if they signed up for the term deposit) based on the other columns.

The best performing model was a VotingEnsemble which combined the outputs of several different models, and was found from AutoML.

## Scikit-learn Pipeline
The pipeline consists of loading the data from a URL, then cleaning it, and fitting a model to it.  Hyperparameters were tuned using AzureML's HyperDrive by randomly searching a space.  The algorithm was a logistic regression model since this is binary classification, and the `C` regularization hyperparameter was optimized.  The random sampler can be good for checking a wide variety of values to optimize the model.  The early stopping model, bandit, cuts off runs if they are not improving their accuracy enough and can save compute power.

## AutoML
The best model from AutoML was a VotingEnsemble model which combines several models' predictions to come up with final predictions.

## Pipeline comparison
The logistic regression model had an accuracy of 0.9048312697256615, while AutoML had accuracy of 0.9485.  Clearly the AutoML model is superior.  The logistic regression model is matrix math essentially, and the AutoML model is a complex combination of several models.  The AutoML model tries a lot more models and is bound to be better.  The logistic regression model also didn't have enough of the hyperparameter space searched so is probably underperforming a bit.

## Future work
The logistic regression model hyperparameter search could be improved, because we only searched a tiny amount of the hyperparamteer space.  We could use the `choice()` function to choose a few typical values, like 0.1, 1, and 10.

For AutoML, we could enable stacked models, enable neural networks, and the iterations of number of different models the AutoML algorithm is allowed to try.  We should also blacklist algos that take a long time, like SVMs, to be more efficient with time.  The SVM algo in the run took almost half the total time.

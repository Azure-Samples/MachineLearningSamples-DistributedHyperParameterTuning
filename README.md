# Distributed Tuning of Hyperparameters using Azure Machine Learning Workbench

## Link to the Microsoft DOCS site

The detailed documentation for this real world scenario includes the step-by-step walkthrough:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-distributed-tuning-of-hyperparameters](https://docs.microsoft.com/azure/machine-learning/preview/scenario-distributed-tuning-of-hyperparameters).


## Link to the Gallery GitHub repository

The public GitHub repository for this real world scenario contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-DistributedHyperParameterTuning](https://github.com/Azure/MachineLearningSamples-DistributedHyperParameterTuning).

## Overview

Many machine learning algorithms have one or more knobs, called hyperparameters. These knobs allow tuning of algorithms to optimize their performance over future data, measured according to user-specified metrics (for example, accuracy, AUC, RMSE). Data scientist needs to provide values of hyperparameters when building a model over training data and before seeing the future test data. How based on the known training data can we set up the values of hyperparameters so that the model has a good performance over the unknown test data? 

A popular technique for tuning hyperparameters is a *grid search* combined with *cross-validation*. Cross-validation is a technique that assesses how well a model, trained on a training set, predicts over the test set. Using this technique, initially we divide the dataset into K folds and then train the algorithm K times, in a round-robin fashion, on all but one of the folds, called held-out fold. We compute the average value of the metrics of K models over K held-out folds. This average value, called *cross-validated performance estimate*, depends on the values of hyperparameters used when creating K models. When tuning hyperparameters, we search through the space of candidate hyperparameter values to find the ones that optimize cross-validation performance estimate. Grid search is a common search technique, where the space of candidate values of multiple hyperparameters is a cross-product of sets of candidate values of individual hyperparameters.

Grid search using cross-validation can be time-consuming. If an algorithm has 5 hyperparameters, each with 5 candidate values and we use K=5 folds, then to complete a grid search we need to train 5x5x5x5x5x5=15625 models. Fortunately, grid-search using cross-validation is an embarrassingly parallel procedure and all these models can be trained in parallel.

This scenario shows how to use Azure Machine Learning Workbench to scale out tuning of hyperparameters of machine learning algorithms that implement scikit-learn API. We show how to configure and use a remote Docker container and Spark cluster as an execution backend for tuning hyperparameters.

## Key components needed to run this scenario

* Ubuntu Data Science Virtual Machine. We recommend using a virtual machine with at least 8 cores and 28 Gb of memory.
* Spark HDInsight cluster. We recommend having a cluster with at least 4 worker nodes and at least 28 Gb of memory in each node. 
* Azure storage account. 

# Data/Telemetry

This tutorial collects usage data and sends it to Microsoft to help improve our products and services. Read our [privacy statement](http://go.microsoft.com/fwlink/?LinkId=521839) to learn more. 

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

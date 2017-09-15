---
title: Distributed Tuning of Hyperparameters using Azure Machine Learning Workbench | Microsoft Docs
description: This scenario shows how to do distributed tuning of hyperparameters using Azure Machine Learning Workbench
services: machine-learning
author: pechyony

ms.service: machine-learning
ms.topic: article
ms.date: 09/13/2017
ms.author: pechyony

---

# Distributed Tuning of Hyperparameters using Azure Machine Learning Workbench

## Link of the Gallery GitHub repository
Following is the link to the public GitHub repository: 

[https://github.com/Azure/MachineLearningSamples-DistributedHyperParameterTuning](https://github.com/Azure/MachineLearningSamples-DistributedHyperParameterTuning)

## Introduction

This scenario shows how to use Azure Machine Learning Workbench to scale out tuning of hyperparameters of machine learning algorithms that implement scikit-learn API. We show how to configure and use remote docker and Spark cluster as an execution backend for tuning hyperparameters.

## Use case overview

Many machine learning algorithms have one or more knobs, called hyperparameters. These knobs allow tuning of algorithms to optimize their performance over future data, measured according to user-specified metrics (for example, accuracy, AUC, RMSE). Data scientist needs to provide values of hyperparameters when building a model over training data and before seeing the future test data. How based on the known training data can we set up the values of hyperparameters so that the model has a good performance over the unknown test data? 

A popular technique for tuning hyperparameters is a *grid search* combined with *cross-validation*. Cross-validation is a technique that assesses how well a model, trained on a training set, predicts over the test set. Using this technique, initially we divide the dataset into K folds and then train the algorithm K times, in a round-robin fashion, on all but one of the folds, called held-out fold. We compute the average value of the metrics of K models over K held-out folds. This average value, called *cross-validated performance estimate*, depends on the values of hyperparameters used when creating K models. When tuning hyperparameters, we search through the space of candidate hyperparameter values to find the ones that optimize cross-validation performance estimate. Grid search is a common search technique, where the space of candidate values of multiple hyperparameters is a cross-product of sets of candidate values of individual hyperparameters. 

Grid search using cross-validation can be time-consuming. If an algorithm has 5 hyperparameters, each with 5 candidate values and we use K=5 folds, then to complete a grid search we need to train 5<sup>6</sup>=15625 models. Fortunately, grid-search using cross-validation is an embarrassingly parallel procedure and all these models can be trained in parallel.

## Prerequisites

* An [Azure account](https://azure.microsoft.com/en-us/free/) (free trials are available).
* An installed copy of [Azure Machine Learning Workbench](./overview-what-is-azure-ml.md) following the [quick start installation guide](./quick-start-installation.md) to install the program and create a workspace.
* This scenario assumes that you are running Azure ML Workbench on Windows 10 or MacOS with Docker engine locally installed. 
* To run scenario with remote docker, provision Ubuntu Data Science Virtual Machine (DSVM) by following the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-provision-vm). We recommend using a virtual machine with at least 8 cores and 28 Gb of memory.
* To run this scenario with Spark cluster, provision HDInsight cluster by following the instructions [here](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql). We recommend having a cluster with at least four worker nodes and at least 28 Gb of memory in each node. To maximize performance of the cluster, we recommend to change the parameters spark.executor.instances, spark.executor.cores, and spark.executor.memory by following the instructions [here](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-resource-manager) and editing the definitions in "custom spark defaults" section.
* Create Azure storage account that is used for storing dataset. You can find instructions for creating storage account [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-create-storage-account).

## Data description

We use [TalkingData dataset](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data). This dataset has events from the apps in cell phones. The goal is to predict gender and age category of cell phone user given the type of the phone and the events that the user generated recently.  

## Scenario structure
This scenario has multiple folders in GitHub repository. Code and configuration files are in **Code** folder, all documentation is in **Docs** folder and all images are **Images** folder. The root folder has README file that contains a brief summary of this scenario.

### Configuration of execution environments
We run our code in remote Docker and in Spark. We also use [scikit-learn](https://anaconda.org/conda-forge/scikit-learn), [xgboost](https://anaconda.org/conda-forge/xgboost), and [azure-storage](https://pypi.python.org/pypi/azure-storage) packages that are not provided in the default Docker container of Azure Machine Learning Workbench. azure-storage package requires installation of [cryptography](https://pypi.python.org/pypi/cryptography) and [azure](https://pypi.python.org/pypi/azure) packages. To install these packages in Docker image and Spark we modify conda_dependencies.yml file:

    name: project_environment
    channels:
      - conda-forge
    dependencies:
      - python=3.5.2
      - scikit-learn
      - xgboost
      - pip:
        - cryptography
        - azure
        - azure-storage

We also use spark-sklearn package to use Spark for distributed tuning of hyperparameters. We modified spark_dependencies.yml file to install this package when Spark execution environment is used:

    configuration: {}
    repositories:
      - "https://mmlspark.azureedge.net/maven"
      - "https://spark-packages.org/packages"
    packages:
      - group: "com.microsoft.ml.spark"
        artifact: "mmlspark_2.11"
        version: "0.7"
      - group: "databricks"
        artifact: "spark-sklearn"
        version: "0.2.0"

In the next steps, we create remote docker and Spark execution environments. Open command line window (CLI) by clicking File menu in the top left corner of AML Workbench and choosing "Open Command Prompt." Then run in CLI

    az login

You get a message

    To sign in, use a web browser to open the page https://aka.ms/devicelogin and enter the code <code> to authenticate.

Go to this web page, enter the code and sign into your Azure account. After this step, run in CLI

    az account list -o table

and find the subscription ID of Azure subscription that has your AML Workbench Workspace account. Finally, run in CLI

    az account set -s <subscription ID>

to complete the connection to your Azure subscription. At this point, we are ready to create remote docker and Spark environments. To set up a remote docker environment, run in CLI

    az ml computetarget attach --name dsvm --address <IP address> --username <username> --password <password> --type remotedocker

with IP address, user name and password in DSVM. IP address of DSVM can be found in Overview section of your DSVM page in Azure portal:
![VM IP](media/scenario-distributed-tuning-of-hyperparameters/vm_ip.png)
To set up Spark environment, run in CLI

    az ml computetarget attach --name spark --address <cluster name>-ssh.azurehdinsight.net  --username <username> --password <password> --type cluster

with the name of the cluster, cluster's SSH user name and password. The default value of SSH user name is `sshuser`, unless you changed it during provisioning of the cluster. The name of the cluster can be found in Properties section of your cluster page in Azure portal:

![Cluster name](../Images/cluster_name.png)

### Data ingestion
The code in this scenario assumes that the data is stored in Azure blob storage. We show initially how to download data from Kaggle site to your computer and upload it to the blob storage. Then we show how to read the data from blob storage. 

To download data from Kaggle, go to [dataset page](https://www.kaggle.com/c/talkingdata-mobile-user-demographics/data) and click Download button. You will be asked to log in to Kaggle. After logging in, you will be redirected back to dataset page. Then download each file in the right column by selecting it and clicking Download button. The total size of seven files in the dataset is 289 Mb. To upload these files to blob storage, create blob storage container 'dataset' in your storage account. You can do that by going to Azure page of your storage account, clicking Blobs and then clicking +Container. Enter 'dataset' as Name and click OK. The following screenshots illustrate these steps:

![Open blob](../Images/open_blob.png)
![Open container](../Images/open_container.png)

After that select dataset container from the list and click Upload button. Azure portal allows to upload multiple files concurrently. In "Upload blob" section click folder button, select all files from the dataset, click Open, and then click Upload. The screenshot below illustrates these steps:

![Upload blob](../Images/upload_blob.png) 

Upload of the files takes several minutes, depending on your Internet connection. 

In our code, we use [Azure Storage SDK](https://azure-storage.readthedocs.io/en/latest/) to download the dataset from blob storage to the current execution environment. The download is performed in load\_data() function from load_data.py file. To use this code, you need to replace <ACCOUNT_NAME> and <ACCOUNT_KEY> with the name and primary key of your storage account that hosts the dataset. Account name is shown in the top left corner of Azure page of your storage account. To get account key, select Access Keys in Azure page of storage account (see the first screenshot in Data Ingestion section) and then copy the long string in the first row of key column:
 
![access key](../Images/access_key.png)

The following code from load_data() function downloads a single file:

    from azure.storage.blob import BlockBlobService

    # Define storage parameters 
    ACCOUNT_NAME = "<ACCOUNT_NAME>"
    ACCOUNT_KEY = "<ACCOUNT_KEY>"
    CONTAINER_NAME = "dataset"

    # Define blob service     
    my_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

    # Load blob
    my_service.get_blob_to_path(CONTAINER_NAME, 'app_events.csv.zip', 'app_events.csv.zip')

### Feature engineering
The code for computing all features is in feature_engineering.py file. We create multiple feature sets:
* One-hot encoding of brand and model of the cell phone (one\_hot\_brand_model function)
* Fraction of events generated by user in each weekday (weekday\_hour_features function)
* Fraction of events generated by user in each hour (weekday\_hour_features function)
* Fraction of events generated by user in each combination of weekday and hour (weekday\_hour_features function)
* Fraction of events generated by user in each app (one\_hot\_app_labels function)
* Fraction of events generated by user in each app label (one\_hot\_app_labels function)
* Fraction of events generated by user in each app category (text\_category_features function)
* Indicator features for categories of apps that were used by used to generated events (one\_hot_category function)

These features were inspired by Kaggle kernel [A linear model on apps and labels](https://www.kaggle.com/dvasyukova/a-linear-model-on-apps-and-labels).

The computation of these features requires significant amount of memory. Initially we tried to compute features in the local environment with 16 Gb RAM. We were able to compute the first four sets of features, but received 'Out of memory' error when computing the fifth feature set. The computation of the first four feature sets is in singleVMsmall.py file and it can be executed in the local environment by running 

     az ml experiment submit -c local .\singleVMsmall.py   

in CLI window.

Since local environment is too small for computing all feature sets, we switch to remote DSVM that has larger memory. The execution inside DSVM is done inside Docker container that is managed by AML Workbench. Using this DSVM we are able to compute all features and train models and tune hyperparameters (see the next section). singleVM.py file has complete feature computation and modeling code. In the next section, we will show how to run singleVM.py in remote DSVM. 

### Tuning hyperparameters using remote DSVM
We use [xgboost](https://anaconda.org/conda-forge/xgboost) implementation [1] of gradient tree boosting. We use [scikit-learn](http://scikit-learn.org/) package to tune hyperparameters of xgboost. Although xgboost is not part of scikit-learn package, it implements scikit-learn API and hence can be used together with hyperparameter tuning functions of scikit-learn. 

Xgboost has eight hyperparameters:
* n_esitmators
* max_depth
* reg_alpha
* reg_lambda
* colsample\_by_tree
* learning_rate
* colsample\_by_level
* subsample
* objective
A description of these hyperparameters can be found [here](http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) and [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md). 
Initially we use remote DSVM and tune hyperparameters from a small grid of candidate values:

    tuned_parameters = [{'n_estimators': [400,500], 'max_depth': [4], 'objective': ['multi:softprob'], 'reg_alpha': [1], 'reg_lambda': [1], 'colsample_bytree': [1],'learning_rate': [0.1], 'colsample_bylevel': [0.1,1], 'subsample': [0.5,1]}]  

This grid has eight combinations of values of hyperparameters. We use 5-fold cross validation, resulting 8x5=40 runs of xgboost. To measure performance of the models, we use negative log loss metric. The following code finds the values of hyperparameters from the grid that maximize the cross-validated negative log loss. The code also uses these values to train the final model over the full training set:

    clf = XGBClassifier(seed=0)
    metric = 'neg_log_loss'
    
    clf_cv = GridSearchCV(clf, tuned_parameters, scoring=metric, cv=5, n_jobs=8)
    model = clf_cv.fit(X_train,y_train)

After creating the model, we save the results of the hyperparameter tuning. We use logging API of AML Workbench to save the best values of hyperparameters and corresponding cross-validated estimate of the negative log loss:

    from azureml.logging import get_azureml_logger

    # initialize logger
    run_logger = get_azureml_logger()

    ...

    run_logger.log(metric, float(clf_cv.best_score_))

    for key in clf_cv.best_params_.keys():
        run_logger.log(key, clf_cv.best_params_[key]) 

We also create sweeping_results.txt file with cross-validated negative log losses of all combinations of hyperparameter values in the grid:

    if not path.exists('./outputs'):
        makedirs('./outputs')
    outfile = open('./outputs/sweeping_results.txt','w')

    print("metric = ", metric, file=outfile)
    for i in range(len(model.grid_scores_)):
        print(model.grid_scores_[i], file=outfile)
    outfile.close()

This file is stored in a special ./outputs directory. Later on we show how to download it.  

 Before running singleVM.py in remote DSVM for the first time, we create a Docker container there by running 

    az experiment prepare --run-configuration dsvm

in CLI windows. Creation of Docker container takes several minutes. After that we run singleVM.py in DSVM:

    az ml experiment submit -c dsvm .\singleVM.py

The logged values can be viewed in Run History window of AML Workbench:

![run history](../Images/run_history.png)

By default Run History window shows values and graphs of the first 1-2 logged values. To see the full list of the chosen values of hyperparameters, click on the settings icon marked with red circle in the above screenshot and select the hyperparameters to be shown in the table. Also, to select the graphs that are shown in the top part of Run History window, click on the setting icon marked with blue circle and select the graphs from the list. 

The chosen values of hyperparameters can also be examined in Run Properties window: 

![run properties](../Images/run_properties.png)

In the top right corner of Run Properties window there is a section Output Files with the list of all files that were created in '.\output' folder in the execution environment. sweeping\_results.txt can be downloaded from there by selecting it and clicking Download button. sweeping_results.txt should have the following output:

    metric =  neg_log_loss
    mean: -2.28712, std: 0.03822, params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'subsample': 0.5, 'n_estimators': 400, 'reg_alpha': 1, 'objective': 'multi:softprob', 'colsample_bylevel': 0.1, 'reg_lambda': 1, 'max_depth': 3}
    mean: -2.28512, std: 0.03861, params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'subsample': 0.5, 'n_estimators': 500, 'reg_alpha': 1, 'objective': 'multi:softprob', 'colsample_bylevel': 0.1, 'reg_lambda': 1, 'max_depth': 3}
    mean: -2.28530, std: 0.03927, params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'subsample': 0.5, 'n_estimators': 400, 'reg_alpha': 1, 'objective': 'multi:softprob', 'colsample_bylevel': 0.1, 'reg_lambda': 1, 'max_depth': 4}
    mean: -2.28513, std: 0.03986, params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'subsample': 0.5, 'n_estimators': 500, 'reg_alpha': 1, 'objective': 'multi:softprob', 'colsample_bylevel': 0.1, 'reg_lambda': 1, 'max_depth': 4}

### Tuning hyperparameters using Spark cluster
We use Spark cluster to scale out tuning hyperparameters and use larger grid. Our new grid is

    tuned_parameters = [{'n_estimators': [300,400,500,600,700], 'max_depth': [4], 'objective': ['multi:softprob'], 'reg_alpha': [1], 'reg_lambda': [1], 'colsample_bytree': [1], 'learning_rate': [0.1], 'colsample_bylevel': [0.01, 0.1, 0.5, 1], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}]

This grid has 120 combinations of values of hyperparameters. Since we use 5-fold cross validation, we run xgboost 120x5=600 times.

scikit-learn package does not have a native support of tuning hyperparameters using Spark cluster. Fortunately, [spark-sklearn](https://spark-packages.org/package/databricks/spark-sklearn) package from Databricks fills this gap. This package provides GridSearchCV function that has almost the same API as GridSearchCV function in scikit-learn. To use spark-sklearn and tune hyperparameters using Spark we need to connect to create Spark context

    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()

Then we replace 

    from sklearn.model_selection import GridSearchCV

with 

    from spark_sklearn import GridSearchCV

Also we replace the call to GridSearchCV from scikit-learn to the one from spark-sklearn:

    clf_cv = GridSearchCV(sc = sc, param_grid = tuned_parameters, estimator = clf, scoring=metric, cv=5)

The final code for tuning hyperparameters using Spark is in distributed\_sweep.py file. The difference between singleVM.py and distributed_sweep.py is in definition of grid and additional four lines of code. Notice also that due to AML Workbench services, the logging code does not change when changing execution environment from remote DSVM to Spark cluster.

Before running distributed_sweep.py in Spark cluster for the first time, we need to install Python packages there. This can be achieved by running 

    az experiment prepare --run-configuration spark

in CLI windows. This installation takes several minutes. After that we run distributed_sweep.py in Spark cluster:

    az ml experiment submit -c spark .\distributed_sweep.py

The results of tuning hyperparameters in Spark cluster, namely logs, best values of hyperparameters and sweeping_results.txt file, can be accessed in Azure Machine Learning Workbench in the same way as in remote DSVM execution. 

### Architecture diagram

The following diagram shows end-to-end workflow:
![architecture](../Images/architecture.png) 

## Conclusion 

In this scenario, we showed how to use Azure Machine Learning Workbench to perform tuning of hyperparameter in remote virtual machine and in remote Spark cluster. We saw that Azure Machine Learning Workbench provides tools for easy configuration of execution environments and switching between them. 

## References

[1] T. Chen and C. Guestrin. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). KDD 2016.





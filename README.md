# AAD Titanic ML Group 3

Objective:
> Create a pareto frontier of ML models to predict if passengers on the titanic survived or not

Team:
* Tristan Peat
* Nathan Miao
* Rishi G
* Pujith

I do not think it will be difficult to create a winning model (bc this project has been done so many times), so we can just do the preprocessing code, then create a process to write the winning model params to a dictionary or .csv and pickle the best models on the pareto frontier (so we can use them without retraining)

## Models
I say we start this with list and expand if we can do better

* Linear regression
* Logistic regression
* knn
* svm
* naive bayes
* decision tree
* random forest
* xgboost

## Ideas:
* I think the name has a lot of value: it is harder to parse these strings, but we can extract information about title and family and age (when age is missing)

## Hyper-parameter tuning
* Could just grid search for all of them

## Saving details about model
* I say we `pickle` all of the models
* IF we are doing thousands of model iterations, then maybe we could only keep the models on the pareto front

## Best Model
![Tree](tree.pdf)

## TODO
* slides

## References
* [Titanic on Kaggle](https://www.kaggle.com/c/titanic)
* [20 popular modles](https://www.kaggle.com/code/vbmokin/titanic-0-83253-comparison-20-popular-models/notebook)

# ML2223

## About

This project is a collaborative effort by António Cardoso, Bárbara Santos, and Isabel Brito as part of the Machine Learning I university course under the Artificial Intelligence and Data Science Bachelor's program at the University of Porto.

The primary aim of this project is to select a machine learning algorithm, modify its functionalities, and create new models to improve its performance under specific conditions.

## Project Objective

For this project, our group has decided to modify the AdaBoost algorithm. The used implementation of AdaBoost was adapted from the one that can be found [here](https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50). Our goal is to achieve superior results specifically for binary datasets with significant label imbalance.

## Changes Made to the Original Algorithm

The group applied 2 modifications to the original algorithm and also analysed their behaviour together.

The first modified algorithm uses a different approach when calculating the alpha value, which is used for updating sample weights. In this model, we used an alpha directly proportional to the error.

The second one, during each iteration of the classifier, duplicates misclassified samples with a probability P. The value of P is determined by dividing the number of newly misclassified samples by the total size of the samples, including the ones previously added.

## Methodology

#### Step 1: Evaluate our implementation of an AdaBoost Classifier for each dataset

- Run 10-fold CV and check the mean accuracy
- Obtain other metrics from a Confusion Matrix
- Get the Learning Curve, ROC Curve and AUC Score

#### Step 2: Define the modified algorithms

#### Step 3: Performance comparison of the modified algorithms with the base implementation on the various datasets

- Acquire each models’ mean accuracy from a 10-fold CV run for each dataset
- Calculate the models’ ranks by the obtained accuracies
- Perform hypothesis tests: Friedman and Nemenyi Post-Hoc tests

## Results

| Dataset | Base Model | Alpha Changed  | Misclassified Duplicated | Both |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Dataset 15 | 0.96  (2) | 0.95  (4) | 0.96  (3) | 0.97  (1) |
| Dataset 24 | 0.92  (2) | 0.91  (3) | 0.93  (1) | 0.91  (4) |
| Dataset 3904 | 0.81  (1) | 0.80  (2) | 0.75  (3) | 0.62  (4) |
| Dataset 146820 | 0.95  (2) | 0.95  (1) | 0.94  (3) | 0.94  (4) | 
| Average Rank | 1.75 | 2.50 | 2.50 | 3.25 |

Label: Mean Accuracy of a 10-fold CV (rank)

After obtaining the results presented in the previous table, we performed a hypothesis test to check whether all algorithms could be considered equivalent, by evaluating the similarity between their rankings across the tasks.

With this in mind, we realized a Friedman test, which gave us a p-value of 0.44.

Considering a significance level of 5%, we can certainly accept the null hypothesis, i.e., the test concludes that there is no significant difference in performance among the algorithms.

<sub><sup>README.md by Bárbara Santos</sup></sub>

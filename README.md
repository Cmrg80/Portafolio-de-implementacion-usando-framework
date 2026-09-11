# Random Forest - Breast Cancer Classification

## Description

Implementation of a Random Forest classifier using
Scikit-learn for the classification of breast tumors.

## Dataset

Breast Cancer Wisconsin Diagnostic Dataset.

Target variable:
- B: Benign
- M: Malignant

The ID column was excluded from the model.

## Model

Random Forest Classifier

Main configuration:

- 100 trees
- Gini criterion
- sqrt feature selection
- Bootstrap enabled
- Random state: 42

## Train/Test Split

80% training
20% testing

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Installation

pip install -r requirements.txt

## Execution

python src/random_forest.py

## Results

The results are available in:

results/

## Report

The complete analysis is available in:

report/reporte_random_forest.pdf

# Decision Tree Classifier

A simple Decision Tree Classifier implemented in Python using scikit-learn, trained on the classic Iris dataset.

**Author:** Onkar Jedhe

## Overview

This project demonstrates how to:
- Load and split a dataset for training/testing
- Train a `DecisionTreeClassifier` from scikit-learn
- Evaluate performance with accuracy and a classification report
- Visualize results with a confusion matrix and a plotted decision tree

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python decision_tree_classifier.py
```

This will:
1. Train the model on the Iris dataset (80/20 train-test split)
2. Print accuracy and a classification report to the console
3. Save `confusion_matrix.png` and `decision_tree.png` in the project folder

## Sample Output

```
Accuracy: 96.67%

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

## Files

| File | Description |
|---|---|
| `decision_tree_classifier.py` | Main script: training, evaluation, visualization |
| `requirements.txt` | Python dependencies |
| `confusion_matrix.png` | Generated confusion matrix (after running the script) |
| `decision_tree.png` | Generated decision tree plot (after running the script) |

## License

This project is open source and available under the MIT License.

"""
Decision Tree Classifier
Author: Onkar Jedhe

A simple, well-documented Decision Tree Classifier built using
scikit-learn on the classic Iris dataset. Includes training,
evaluation, a confusion matrix plot, and a visualization of the
trained tree.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def load_data():
    """Load the Iris dataset and return features, labels, and names."""
    data = load_iris()
    return data.data, data.target, data.feature_names, data.target_names


def train_model(X_train, y_train, max_depth=3, random_state=42):
    """Train a DecisionTreeClassifier on the given training data."""
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Print accuracy and classification report; return predictions."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return y_pred


def plot_confusion(y_test, y_pred, target_names, out_path="confusion_matrix.png"):
    """Save a confusion matrix plot to disk."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Decision Tree Classifier")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def plot_decision_tree(model, feature_names, target_names, out_path="decision_tree.png"):
    """Save a visualization of the trained decision tree to disk."""
    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Decision tree plot saved to {out_path}")


def main():
    X, y, feature_names, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train, max_depth=3)

    y_pred = evaluate_model(model, X_test, y_test, target_names)

    plot_confusion(y_test, y_pred, target_names)
    plot_decision_tree(model, feature_names, target_names)


if __name__ == "__main__":
    main()

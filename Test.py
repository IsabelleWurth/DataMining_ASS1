import pandas as pd
import numpy as np
from DecisionTree import DecisionTree, tree_pred
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def pre_process(data):
    data = pd.DataFrame(data)
    data = data.astype(int)
    target_column = data[-1]
    classification = data[target_column].to_numpy()
    features = data.drop(target_column, axis=1).to_numpy()
    return features, classification

def pre_process_indian(data):
    data = pd.DataFrame(data, columns = ['zero','one','two','three','four','five', 'six', 'seven', 'class'])
    int_columns = ['zero', 'one', 'two', 'three', 'four', 'seven', 'class']  # Columns to be converted to int
    data[int_columns] = data[int_columns].astype(int)

    # Convert specific columns to floats
    float_columns = ['five', 'six']  # Columns to be converted to float
    data[float_columns] = data[float_columns].astype(float)
    classification = data['class'].to_numpy()
    features = data.drop('class', axis=1).to_numpy()
    return features, classification

def evaluate(real_class, predictions):
    cm = confusion_matrix(real_class, predictions)
    
    # Create a DataFrame for better readability
    cm_df = pd.DataFrame(cm)
    return cm_df



if __name__ == "__main__":
    # Load dataset
    data = data = np.genfromtxt('indian.txt', delimiter=',')

    # Pre-process data to optain features (x) and labels (y)
    x, y = pre_process_indian(data)

    # Initiate DecisionTree with nmin, minleaf, en nfeat
    tree = DecisionTree(nmin=20, minleaf=5, nfeat=8)

    # Fit the decision tree 
    tree.fit(x, y)

    # Nieuwe data waarvoor we voorspellingen willen doen
    new_data = x
    new_data_predictions = y
    
    # Je zou nu de tree structuur kunnen visualiseren of traverseren om te zien of de boom goed is gegroeid
    # def traverse_tree(node, depth=0):
    #     """Recursively traverse the tree to print its structure."""
    #     if node.is_terminal():
    #         print(f"{'  '*depth}Leaf node with class: {node.c_label}, Gini: {node.gini}")
    #     else:
    #         print(f"{'  '*depth}Split on feature {node.split_feature} with threshold {node.split_value}, Gini: {node.gini}")
    #         traverse_tree(node.left, depth + 1)
    #         traverse_tree(node.right, depth + 1)

    # # Visualiseer de boom
    # traverse_tree(tree.root)

    # Voorspellingen maken met de getrainde boom
    predictions = tree_pred(new_data, tree.root)
    evaluations = evaluate(new_data_predictions, predictions)
    #print("Predicted class labels:", predictions)
    print(evaluations)
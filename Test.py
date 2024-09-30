import pandas as pd
import numpy as np
from DecisionTree import DecisionTree, tree_pred, tree_pred_b

def pre_process(data):
    data = pd.DataFrame(data, columns=['age', 'married', 'house', 'income', 'gender', 'class'])
    data = data.astype(int)
    classification = data['class'].to_numpy()
    features = data.drop('class', axis=1).to_numpy()
    return features, classification

if __name__ == "__main__":
    # Load dataset
    data = np.array([
        [22, 0, 0, 28, 1, 0],  # Features: age, married, house, income, gender
        [46, 0, 1, 32, 0, 0],
        [24, 1, 1, 24, 1, 0],
        [25, 0, 0, 27, 1, 0],
        [29, 1, 1, 32, 0, 0],
        [45, 1, 1, 30, 0, 1],
        [63, 1, 1, 58, 1, 1],
        [36, 1, 0, 52, 1, 1],
        [23, 0, 1, 40, 0, 1],
        [50, 1, 1, 28, 0, 1]
    ])

    # Pre-process data to optain features (x) and labels (y)
    x, y = pre_process(data)

    # Initiate DecisionTree with nmin, minleaf, en nfeat
    tree = DecisionTree(nmin=2, minleaf=1, nfeat=5)

    # Fit the decision tree 
    tree.fit(x, y)

    # Nieuwe data waarvoor we voorspellingen willen doen
    new_data = np.array([[30, 1, 1, 40, 1],
                         [55, 0, 1, 30, 0]])
    
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
    trees = tree.tree_grow_b(x, y, m=3)
    y = tree_pred_b(new_data, trees)

    #print("Predicted class labels:", predictions)

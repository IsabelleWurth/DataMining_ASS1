import pandas as pd
import numpy as np
from DecisionTree import DecisionTree, tree_pred, tree_pred_b
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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

def eclipse(file):
    df = pd.read_csv(file, sep = ';')

    # dit zijn vgm de predictors die we moeten gebruiken (zie tabel uit artikel). Het zijn er ook 41 dus dat klopt 
    predictors = ['pre','FOUT_avg', 'FOUT_max', 'FOUT_sum', 'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum', 'NOF_avg', 'NOF_max', 'NOF_sum', 'NOM_avg', 'NOM_max', 'NOM_sum', 'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum', 'PAR_avg', 'PAR_max', 'PAR_sum', 'VG_avg', 'VG_max', 'VG_sum', 'NOCU', 'ACD_avg', 'ACD_max', 'ACD_sum', 'NOI_avg', 'NOI_max', 'NOI_sum', 'NOT_avg', 'NOT_max', 'NOT_sum', 'TLOC_avg', 'TLOC_max', 'TLOC_sum']
    target = 'post'
    df[target] = df[target].astype(int)

    # Change target variables such that it indicates whether post bugs are found or not (instead of how many)
    df.loc[df[target] > 0, target] = 1
    variables = df[predictors].to_numpy()
    target = df[target].to_numpy()
    return variables, target


def evaluate(real_class, predictions):
    cm = confusion_matrix(real_class, predictions) 
    cm_df = pd.DataFrame(cm)
    return cm_df

def metrics(y_test_labels, y_pred_labels): 
    # Bereken de metrics voor de Decision Tree (1=pos en 0=neg)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)

    print("Decision Tree Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}\n")

    return accuracy, precision, recall


if __name__ == "__main__":
    # Load dataset
    x_train, y_train = eclipse('eclipse_train.csv')
    x_test, y_test = eclipse('eclipse_test.csv')

    # Initiate DecisionTree with nmin, minleaf, en nfeat
    tree = DecisionTree(nmin=15, minleaf=5, nfeat=41)

    # Fit the decision tree 
    tree.fit(x_train, y_train)
    
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
    predictions = tree_pred(x_test, tree.root)
    print("Predicted class labels:", predictions)
    metrics = metrics(y_test, predictions)
    print("metrics: ", metrics)
    evaluations = evaluate(y_test, predictions)
    print(evaluations)
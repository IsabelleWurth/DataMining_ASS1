"""
    Created by: 
        - Renske Bos: 2483947
        - Juul Petit: 6446981
        - Vincent Huf: 4546202
        - Isabelle Wurth: 6358500
"""

import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

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

    # 41 features as predictors and the 'post' feature to predict 
    predictors = ['pre','FOUT_avg', 'FOUT_max', 'FOUT_sum', 'MLOC_avg', 'MLOC_max', 'MLOC_sum', 'NBD_avg', 'NBD_max', 'NBD_sum', 'NOF_avg', 'NOF_max', 'NOF_sum', 'NOM_avg', 'NOM_max', 'NOM_sum', 'NSF_avg', 'NSF_max', 'NSF_sum', 'NSM_avg', 'NSM_max', 'NSM_sum', 'PAR_avg', 'PAR_max', 'PAR_sum', 'VG_avg', 'VG_max', 'VG_sum', 'NOCU', 'ACD_avg', 'ACD_max', 'ACD_sum', 'NOI_avg', 'NOI_max', 'NOI_sum', 'NOT_avg', 'NOT_max', 'NOT_sum', 'TLOC_avg', 'TLOC_max', 'TLOC_sum']
    target = 'post'
    df[target] = df[target].astype(int)

    # Change target variables such that it indicates whether post bugs are found or not (instead of how many)
    df.loc[df[target] > 0, target] = 1
    variables = df[predictors].to_numpy()
    target = df[target].to_numpy()
    return variables, target

def evaluate(real_class, predictions):
    # Make a confusion matrix
    cm = confusion_matrix(real_class, predictions) 
    cm_df = pd.DataFrame(cm)
    return cm_df

def metrics(y_test_labels, y_pred_labels): 
    # Calculate metrics for the Decision Tree (1=pos en 0=neg)
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels)
    recall = recall_score(y_test_labels, y_pred_labels)

    # print("Decision Tree Metrics:")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}\n")

    return accuracy, precision, recall


if __name__ == "__main__":
    # Load dataset
    x_train, y_train = eclipse('eclipse_train.csv')
    x_test, y_test = eclipse('eclipse_test.csv')

    # Initiate DecisionTree
    single_tree = DecisionTree()
    # bagging_tree = DecisionTree()
    # rf_tree = DecisionTree()

    # Grow the decision tree (single tree, bagging and random forest)
    single_tree.root = single_tree.tree_grow(x_train, y_train, nmin=20, minleaf=5, nfeat=41)
    # bagging_trees = bagging_tree.tree_grow_b(x_train, y_train, m=100, nmin=15, minleaf=5, nfeat=41)
    # rf_trees = rf_tree.tree_grow_b(x_train, y_train, m=100, nmin=15, minleaf=5, nfeat=6)

    # Create predictions using the trained tree and evaluate (single tree)
    predictions_single = single_tree.tree_pred(x_test, single_tree.root)
    # Create predictions using the trained tree and evaluate (bagging)
    # predictions_bagging = bagging_tree.tree_pred_b(x_test, bagging_trees)
    # Create predictions using the trained tree and evaluate (random forest)
    # predictions_rf = rf_tree.tree_pred_b(x_test, rf_trees)
        
    # Calculate metrics for this run - TEST SINGLE 
    single_tree_metrics_run = metrics(y_test, predictions_single)
    single_tree_conf_matrix = evaluate(y_test, predictions_single)
        
    # Calculate metrics for this run - TEST BAGGING
    # bagging_metrics_run = metrics(y_test, predictions_bagging)
    # bagging_conf_matrix = evaluate(y_test, predictions_bagging)

    # Calculate metrics for this run - TEST RANDOM FOREST
    # rf_metrics_run = metrics(y_test, predictions_rf)
    # rf_conf_matrix = evaluate(y_test, predictions_rf)

       
    # Summarize and print the results for each model
    print("\nMetrics for Single Tree:")
    print(f"Accuracy = {single_tree_metrics_run[0]}, Precision = {single_tree_metrics_run[1]}, Recall = {single_tree_metrics_run[2]}")
    
    # print("\nMetrics for Bagging:")
    # print(f"Accuracy = {bagging_metrics_run[0]}, Precision = {bagging_metrics_run[1]}, Recall = {bagging_metrics_run[2]}")

    # print("\nMetrics for Random Forest:")
    # print(f"Accuracy = {rf_metrics_run[0]}, Precision = {rf_metrics_run[1]}, Recall = {rf_metrics_run[2]}")

    # Print the confusion matrices if needed
    print(f"Confusion Matrix for Single Tree Run \n{single_tree_conf_matrix}")
    # print(f"Confusion Matrix for Bagging: \n{bagging_conf_matrix}")
    # print(f"Confusion Matrix for Random Forest: \n{rf_conf_matrix}")
    




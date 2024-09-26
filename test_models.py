import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, LSTM, TimeDistributed, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorboard.plugins.hparams import api as hp
import joblib
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Metric
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from plot_training_results import plot_training_results
from plot_confusion_matrix import plot_confusion_matrix
from plot_example_data import plot_example_data
import datetime
import csv


def load_data():
    # To load normalized folds and scalers
    normalized_data = joblib.load('normalized_data_Softness_Train_Test.pkl')
    encoder = joblib.load('softnessencoder__Softness_Train_Test.pkl')

    # normalized_folds = joblib.load('normalized_folds_pca_allData_peakTrim_justsoftness.pkl')
    scalers = joblib.load('scalers_pca__Softness_Train_Test.pkl')
    testscalers = joblib.load('testscalers_pca__Softness_Train_Test.pkl')
    # encoded_softness = joblib.load('encoded_softness_pca_allData_peakTrim_justsoftness.pkl')
    # encoder = joblib.load('softnessencoder_pca_allData_peakTrim_justsoftness.pkl')

    #plot_example_data(normalized_folds)

    # Define window size and number of classes
    window_size = normalized_data[0][0].shape[0]  # Assuming all windows have the same size
    num_classes = 4#encoded_softness.shape[1]

    return normalized_data, window_size, num_classes, encoder

def time_divide_data(normalized_data):
    test_windows = normalized_data[2]
    win_size = 10
    
    for j, win in enumerate(test_windows):
        startIdx = 0
        stopIdx = win_size
        new_win = []
        while stopIdx <= win.shape[0]:
            new_win.append(win[startIdx:stopIdx, :])
            startIdx = int(startIdx+(win_size/2))
            stopIdx = int(stopIdx+(win_size/2))
        test_windows[j] = np.array(new_win)
        
    return normalized_data

def run_trial():
    # Lists to store results
    accuracy_scores = []
    rec_scores = []
    prec_scores = []
    f1_int_scores = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []

    normalized_data, window_size, num_classes, encoder = load_data()
    time_div_data = time_divide_data(normalized_data)

    test_windows = np.array(time_div_data[2])
    test_labels = np.array(time_div_data[3])
    test_labels = test_labels[:, 1]
    test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))
    model = load_model('Final Softness Results - Properly/SoftnessModel.keras')

    # Evaluate on Test data
    print(f"Input test data shape: {test_windows.shape}")
    test_loss, test_accuracy, test_prec, test_rec, test_f1_int = model.evaluate(test_windows, test_labels_encoded, verbose=0)
    y_test_pred = model.predict(test_windows)
    y_test_true = np.argmax(test_labels_encoded, axis=1)
    y_test_pred = np.argmax(y_test_pred, axis=1)

    # Accumulate predictions and true labels for confusion matrix
    all_y_true.append(y_test_true)
    all_y_pred.append(y_test_pred)

    # Calculate F1 score for test
    f1 = f1_score(y_test_true, y_test_pred, average='macro')

    accuracy_scores.append(test_accuracy)
    rec_scores.append(test_rec)
    prec_scores.append(test_prec)
    f1_scores.append(f1)
    f1_int_scores.append(test_f1_int)


    # Print results
    print(f"Test Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Test F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Test F1 Internal Score: {np.mean(f1_int_scores):.4f} ± {np.std(f1_int_scores):.4f}")
    print(f"Test Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print(f"Test Recall: {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")

    results = {"acc"  : np.mean(accuracy_scores),
               "f1"   : np.mean(f1_scores), 
               "prec" : np.mean(prec_scores), 
               "rec"  : np.mean(rec_scores), 
               "yTrue": all_y_true, 
               "yPred": all_y_pred}

    return results, encoder.categories_


if __name__ == "__main__":

    results, categories = run_trial()

    # Plot confusion matrix
    plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories)

    # hparam_hist = []
    # hparam_hist = [["trial"] + [hprm for hprm in hparams.keys()]]
    # hparam_hist.append([1] + [hprm for hprm in hparams.values()])
    #save_results(results, folds2Test, use_pca, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, 1)

    print("Done")
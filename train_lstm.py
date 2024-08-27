import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
import csv
import datetime


def save_results(results, folds2Test, use_pca, startTimeStamp, hparam_hist, session_num=1):
    hist = {'loss': [],
            'accuracy': [],
            'f1_score': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_score': []}
    for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'f1_score', 'val_f1_score']:
        for fold in range(folds2Test):
            hist[metric].append([f'{metric}_{fold+1}_trial{session_num}']+ results["hist"][fold].history[metric])
        
        hist[metric].append([f'average_trial{session_num}'] + list(np.mean([history.history[metric] for history in results["hist"]], axis=0)))
        hist[metric].append([f'std_trial{session_num}'] + list(np.std([history.history[metric] for history in results["hist"]], axis=0)))
        hist[metric].append([])

    for m in ['loss', 'accuracy', 'f1_score']:
        name = f"train_hist_{m}_PCA{use_pca}_LSTM_" + startTimeStamp
        with open(f'{name}.csv', 'a') as out:
            for row in hist[m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')
            
            for row in hist["val_"+m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')

    name = f"trial_details_PCA{use_pca}_LSTM_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(hparam_hist)
    
    name = f"lr_details_PCA{use_pca}_LSTM_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(results['lr_hist'])


def load_data(use_pca=False):
    # To load normalized folds and scalers
    if use_pca:
        normalized_folds = joblib.load('normalized_folds_pca_allData_peakTrim.pkl')
        scalers = joblib.load('scalers_pca_allData_peakTrim.pkl')
        encoded_labels = joblib.load('encoded_labels_pca_allData_peakTrim.pkl')
        encoder = joblib.load('encoder_pca_allData_peakTrim.pkl')
    else:
        normalized_folds = joblib.load('normalized_folds_allData_peakTrim.pkl')
        scalers = joblib.load('scalers_allData_peakTrim.pkl')
        encoded_labels = joblib.load('encoded_labels_allData_peakTrim.pkl')
        encoder = joblib.load('encoder_allData_peakTrim.pkl')

    #plot_example_data(normalized_folds)

    # Define window size and number of classes
    window_size = normalized_folds[0][0][0].shape[0]  # Assuming all windows have the same size
    num_classes = encoded_labels.shape[1]

    return normalized_folds, window_size, num_classes, encoder
    
# Define the LSTM model for multi-class classification
def create_lstm_model(input_shape, num_classes, hparams):
    model = Sequential()
    model.add(LSTM(hparams["HP_LSTM_UNITS"], return_sequences=True, recurrent_dropout=0.2, input_shape=input_shape, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"])))
    model.add(LSTM(hparams["HP_LSTM_UNITS"], return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"])))
    #model.add(Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"])))
    model.add(Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"])))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for multi-class

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tfa.metrics.F1Score(num_classes, average='macro')])
    #print(model.summary())
    return model


def run_trial(hparams, folds, use_pca):
    # Lists to store results
    lr_histories = []
    accuracy_scores = []
    rec_scores = []
    prec_scores = []
    f1_int_scores = []
    f1_scores = []
    fold_histories = []
    all_y_true = []
    all_y_pred = []

    normalized_folds, window_size, num_classes, encoder = load_data(use_pca)

    # Train the model on each fold
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(normalized_folds):
        if i >= folds:
            break
        else:
            print(f'Training on fold {i+1}')

            # Shuffle data
            shuffle_indices = np.random.permutation(len(train_windows))
            train_windows = np.array(train_windows)[shuffle_indices]
            shuffled_labels = np.array(train_labels)[shuffle_indices]

            # train_windows = np.array(train_windows)
            test_windows = np.array(test_windows)
            train_labels_encoded = encoder.transform(shuffled_labels.reshape(-1, 1))
            test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))
            
            model = create_lstm_model((window_size, train_windows.shape[-1]), num_classes, hparams)

            reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', verbose=1, mode='max', factor=0.8)
            
            # Define the Required Callback Function
            lr_hist = []
            # class savelearningrate(tf.keras.callbacks.Callback):
            #     def on_epoch_end(self, epoch, logs={}):
            #         optimizer = self.model.optimizer
            #         lr_hist.append(K.eval(optimizer.lr))
            # savelr = savelearningrate()

            history = model.fit(train_windows, train_labels_encoded,
                    epochs=hparams["HP_EPOCHS"], batch_size=hparams["HP_BATCH"], validation_data=(test_windows, test_labels_encoded), shuffle=True, verbose=0)#, callbacks=[reduce_lr, savelr])
            
            print(lr_hist)

            # Evaluate on Test data
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
            fold_histories.append(history)
            lr_histories.append(lr_hist)

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
               "yPred": all_y_pred, 
               "hist" : fold_histories,
               "lr_hist" : lr_histories}

    return results



if __name__ == "__main__":
    # HP_H_UNITS = hp.HParam('h_units', hp.Discrete([0]))
    # HP_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([0]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([0]))
    # HP_BATCH = hp.HParam('batch', hp.Discrete([0]))
    # HP_LR = hp.HParam('lr', hp.Discrete([0]))
    # HP_L2_LAMBDA = hp.HParam('l2_lambda', hp.Discrete([0]))

    hparams = { "HP_H_UNITS": 100,
                "HP_LSTM_UNITS": 256,
                "HP_EPOCHS": 350,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001 }

    folds2Test = 5
    use_pca = True
    results = run_trial(hparams, folds2Test, use_pca)

    hparam_hist = []
    hparam_hist = [["trial"] + [hprm for hprm in hparams.keys()]]
    hparam_hist.append([1] + [hprm for hprm in hparams.values()])
    save_results(results, folds2Test, use_pca, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, 1)

    # Plot confusion matrix
    #for f in range(len(results["yTrue"])):
    #    plot_confusion_matrix(results["yTrue"][f], results["yPred"][f])
    plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs])

    # Plot average training history across folds
    plot_training_results(results["hist"])
    print("Done")
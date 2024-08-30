import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
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
        name = f"SOFTNESStrain_hist_{m}_PCA{use_pca}_LSTM_" + startTimeStamp
        with open(f'{name}.csv', 'a') as out:
            for row in hist[m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')
            
            for row in hist["val_"+m]:
                for col in row:
                    out.write('{0},'.format(col))
                out.write('\n')

    name = f"SOFTNESStrial_details_PCA{use_pca}_LSTM_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(hparam_hist)
    
    name = f"SOFTNESSlr_details_PCA{use_pca}_LSTM_" + startTimeStamp
    with open(f'{name}.csv', 'a') as out:
        write = csv.writer(out)
        write.writerows(results['lr_hist'])


def load_data(use_pca=False):
    # To load normalized folds and scalers
    if use_pca:
        normalized_folds = joblib.load('normalized_folds_pca_allData_peakTrim_softness.pkl')
        scalers = joblib.load('scalers_pca_allData_peakTrim_softness.pkl')
        encoded_softness = joblib.load('encoded_softness_pca_allData_peakTrim_softness.pkl')
        encoder = joblib.load('softnessencoder_pca_allData_peakTrim_softness.pkl')
    else:
        normalized_folds = joblib.load('normalized_folds_allData_peakTrim_softness.pkl')
        scalers = joblib.load('scalers_allData_peakTrim_softness.pkl')
        encoded_softness = joblib.load('encoded_softness_allData_peakTrim_softness.pkl')
        encoder = joblib.load('softnesssencoder_allData_peakTrim_softness.pkl')

    #plot_example_data(normalized_folds)

    # Define window size and number of classes
    window_size = normalized_folds[0][0][0].shape[0]  # Assuming all windows have the same size
    num_classes = encoded_softness.shape[1]

    return normalized_folds, window_size, num_classes, encoder

def time_divide_data(normalized_folds):
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(normalized_folds):
        win_size = 10
        for j, win in enumerate(train_windows):
            startIdx = 0
            stopIdx = win_size
            new_win = []
            while stopIdx <= win.shape[0]:
                new_win.append(win[startIdx:stopIdx, :])
                startIdx = int(startIdx+(win_size/2))
                stopIdx = int(stopIdx+(win_size/2))
            train_windows[j] = np.array(new_win)
        
        for j, win in enumerate(test_windows):
            startIdx = 0
            stopIdx = win_size
            new_win = []
            while stopIdx <= win.shape[0]:
                new_win.append(win[startIdx:stopIdx, :])
                startIdx = int(startIdx+(win_size/2))
                stopIdx = int(stopIdx+(win_size/2))
            test_windows[j] = np.array(new_win)
        
    return normalized_folds


# Define the LSTM model for multi-class classification
def create_cnnlstm_model(input_size, num_classes, hparams):
    modelInput = Input(shape=input_size)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', input_shape=input_size, padding="same"))(modelInput)
    model = TimeDistributed(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', padding="same"))(model)
    model = TimeDistributed(MaxPooling1D(hparams["HP_POOL"]))(model)
    model = TimeDistributed(Flatten())(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = LSTM(hparams["HP_LSTM_UNITS"], return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"]))(model)
    model = Dropout(0.5)(model)
    model = Dense(num_classes, activation='softmax')(model)  # Output layer with softmax for multi-class

    model = Model(modelInput, model)

    opt = tf.keras.optimizers.Adam(learning_rate=hparams["HP_LR"])

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tfa.metrics.F1Score(num_classes, average='macro')])
    
    print(model.summary())
    return model


def run_trial(hparams, folds, use_pca, verbose=0):
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
    time_div_folds = time_divide_data(normalized_folds)

    # Train the model on each fold
    for i, (train_windows, train_labels, test_windows, test_labels) in enumerate(time_div_folds):
        train_labels = train_labels[:, 1]
        test_labels = test_labels[:, 1]
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
            window_size = train_windows[0].shape
            model = create_cnnlstm_model(window_size, num_classes, hparams)

            # reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', verbose=1, mode='max', factor=0.8)
            
            # # Define the Required Callback Function
            lr_hist = []
            # class savelearningrate(tf.keras.callbacks.Callback):
            #     def on_epoch_end(self, epoch, logs={}):
            #         optimizer = self.model.optimizer
            #         lr_hist.append(K.eval(optimizer.lr))
            # savelr = savelearningrate()
            
            history = model.fit(train_windows, train_labels_encoded, epochs=hparams["HP_EPOCHS"], batch_size=hparams["HP_BATCH"], validation_data=(test_windows, test_labels_encoded), shuffle=True, verbose=verbose)
            
            #print(lr_hist)

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

    return results, encoder.categories_



if __name__ == "__main__":
    hparams = { "HP_H_UNITS": 64,
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 80,
                "HP_BATCH": 64,
                "HP_LR": 0.001,
                "HP_L2_LAMBDA": 0.001,
                "HP_LSTM_UNITS": 64 }

    folds2Test = 5
    use_pca = True
    results, categories = run_trial(hparams, folds2Test, use_pca, verbose=1)

    # Plot confusion matrix
    #for f in range(len(results["yTrue"])):
    #    plot_confusion_matrix(results["yTrue"][f], results["yPred"][f])
    plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs], categories)

    # Plot average training history across folds
    plot_training_results(results["hist"])

    hparam_hist = []
    hparam_hist = [["trial"] + [hprm for hprm in hparams.keys()]]
    hparam_hist.append([1] + [hprm for hprm in hparams.values()])
    save_results(results, folds2Test, use_pca, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), hparam_hist, 1)

    print("Done")
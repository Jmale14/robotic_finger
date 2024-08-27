import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten
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


def load_data(use_pca=False):
    # To load normalized folds and scalers
    if use_pca:
        normalized_folds = joblib.load('normalized_folds_pca_allData.pkl')
        scalers = joblib.load('scalers_pca_allData.pkl')
        encoded_labels = joblib.load('encoded_labels_pca_allData.pkl')
        encoder = joblib.load('encoder_pca_allData.pkl')
    else:
        normalized_folds = joblib.load('normalized_folds.pkl')
        scalers = joblib.load('scalers.pkl')
        encoded_labels = joblib.load('encoded_labels.pkl')
        encoder = joblib.load('encoder.pkl')

    #plot_example_data(normalized_folds)

    # Define window size and number of classes
    window_size = normalized_folds[0][0][0].shape[0]  # Assuming all windows have the same size
    num_classes = encoded_labels.shape[1]

    return normalized_folds, window_size, num_classes, encoder
    
# Define the LSTM model for multi-class classification
def create_cnn_model(input_size, num_classes, hparams):
    model = Sequential()
    model.add(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu', input_shape=input_size))
    model.add(MaxPooling1D(hparams["HP_POOL"]))
    model.add(Conv1D(hparams["HP_FILTERS"], hparams["HP_KERNEL"], strides=1, activation='relu'))
    model.add(MaxPooling1D(hparams["HP_POOL"]))
    model.add(Flatten())
    model.add(Dense(hparams["HP_H_UNITS"], activation='relu', kernel_regularizer=l2(hparams["HP_L2_LAMBDA"])))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax for multi-class

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
            
            model = create_cnn_model((window_size, train_windows.shape[-1]), num_classes, hparams)

            # reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', verbose=1, mode='max', factor=0.8)
            
            # # Define the Required Callback Function
            # lr_hist = []
            # class savelearningrate(tf.keras.callbacks.Callback):
            #     def on_epoch_end(self, epoch, logs={}):
            #         optimizer = self.model.optimizer
            #         lr_hist.append(K.eval(optimizer.lr))
            # savelr = savelearningrate()

            history = model.fit(train_windows, train_labels_encoded,
                    epochs=hparams["HP_EPOCHS"], batch_size=hparams["HP_BATCH"], validation_data=(test_windows, test_labels_encoded), shuffle=True, verbose=verbose)#, callbacks=[reduce_lr, savelr])
            
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
            #lr_histories.append(lr_hist)

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
                "HP_FILTERS": 100,
                "HP_KERNEL": 3,
                "HP_POOL": 5,
                "HP_EPOCHS": 200,
                "HP_BATCH": 64,
                "HP_LR": 0.01,
                "HP_L2_LAMBDA": 0.01 }

    folds2Test = 1
    results = run_trial(hparams, folds2Test, True, verbose=1)

    # Plot confusion matrix
    #for f in range(len(results["yTrue"])):
    #    plot_confusion_matrix(results["yTrue"][f], results["yPred"][f])
    plot_confusion_matrix([x for xs in results["yTrue"] for x in xs], [x for xs in results["yPred"] for x in xs])

    # Plot average training history across folds
    plot_training_results(results["hist"])
    print("Done")
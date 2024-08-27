from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB


def load_data(use_pca=False):
    # To load normalized folds and scalers
    if use_pca:
        normalized_folds = joblib.load('normalized_folds_pca.pkl')
        scalers = joblib.load('scalers_pca.pkl')
        encoded_labels = joblib.load('encoded_labels_pca.pkl')
        encoder = joblib.load('encoder_pca.pkl')
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

if __name__ == "__main__":
    use_pca = True
    folds = 5

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
        #train_labels_encoded = encoder.transform(shuffled_labels.reshape(-1, 1))
        #test_labels_encoded = encoder.transform(np.array(test_labels).reshape(-1, 1))

        train_windows = np.reshape(train_windows, (-1, 450))
        test_windows = np.reshape(test_windows, (-1, 450))
            

        # Define and fit the model
        nb_model = GaussianNB()
        nb_model.fit(train_windows, shuffled_labels)

        # Predict the labels
        y_pred = nb_model.predict(test_windows)

        # Evaluate
        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='weighted')
        print(f"Naive Bayes Accuracy: {accuracy:.4f}")
        print(f"Naive Bayes F1 Score: {f1:.4f}")

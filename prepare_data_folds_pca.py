import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


sampling_freq = 50
seconds_to_trim = 4

def trim_start(df, lines_to_trim):
    return df.iloc[lines_to_trim*seconds_to_trim:]

def load_csv_files(directories):
    data_frames = []
    required_columns = ["accx", "accy", "accz", "gx", "gy", "gz", "pressure"]
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.startswith("EXP") and filename.endswith(".csv"):
                # Extract label from the filename
                label = filename.split('_')[0].replace("EXP", "")
                df = pd.read_csv(os.path.join(directory, filename))
                df = trim_start(df, sampling_freq)  # Remove the first 'lines_to_trim' lines
                df = df[required_columns]  # Only keep the required columns
                df['label'] = label  # Add the label column
                data_frames.append(df)
            if filename.startswith("material_"):
                for fName in os.listdir(directory+"/"+filename):
                    if not fName.endswith("delay100.csv"):
                        if (fName.startswith("M") and fName.endswith(".csv")) or (fName.startswith("EXP") and fName.endswith(".csv")):
                            # Extract label from the filename
                            label = filename.split('_')[1]
                            df = pd.read_csv(os.path.join(directory, filename, fName))
                            df = trim_start(df, sampling_freq)  # Remove the first 'lines_to_trim' lines
                            df = df[required_columns]  # Only keep the required columns
                            df['label'] = label  # Add the label column
                            data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)

def create_windows(data, window_size, overlap=0):
    step_size = int(window_size * (1 - overlap))
    windows = []
    labels = []
    for label in data['label'].unique():
        label_data = data[data['label'] == label]
        for i in range(0, len(label_data) - window_size + 1, step_size):
            windows.append(label_data.iloc[i:i + window_size].drop(columns=['label']).values)
            labels.append(int(label))
    return np.array(windows), np.array(labels)

def split_into_folds(windows, labels, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in kf.split(windows):
        folds.append((windows[train_index], labels[train_index], windows[test_index], labels[test_index]))
        print(np.bincount(labels[train_index]))
        print(np.bincount(labels[test_index]))
    return folds

# def normalize_windows(windows, scaler=None):
#     if scaler is None:
#         scaler = StandardScaler()
#         normalized_windows = [scaler.fit_transform(window) for window in windows]
#         return normalized_windows, scaler
#     else:
#         normalized_windows = [scaler.transform(window) for window in windows]
#         return normalized_windows
def normalize_windows(windows, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        # Stack all windows to fit the scaler on the entire dataset
        all_data = np.vstack(windows)
        scaler.fit(all_data)
    
    # Transform each window using the fitted scaler
    normalized_windows = [scaler.transform(window) for window in windows]
    
    return normalized_windows, scaler

def fit_pca(train_windows):
    pca = PCA(n_components=5)
    all_data = np.vstack(train_windows)
    pca.fit(all_data)
    print(pca.explained_variance_ratio_)
    return pca

def plot_pca(pcas):
    plt.figure(figsize=(10, 6))
    cumulative_explained_variance = []
    for pca in pcas:
        cumulative_explained_variance.append(np.cumsum(pca.explained_variance_ratio_))
        plt.plot(cumulative_explained_variance[-1], marker='o', linestyle='--', color='b')
    print(np.mean(cumulative_explained_variance, axis=0))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA')
    plt.grid(True)
    plt.show()

# Specify the directory containing your CSV files
directory = ['collected_data_previous', 'collected_data']
data = load_csv_files(directory)

# Define the window size
window_size = 2
windows, labels = create_windows(data, window_size*sampling_freq, overlap=0)

# Define the number of folds
n_splits = 5
folds = split_into_folds(windows, labels, n_splits)

normalized_folds = []
scalers = []
pcas = []
for train_windows, train_labels, test_windows, test_labels in folds:
    train_windows, train_scaler = normalize_windows(train_windows)
    pca = fit_pca(train_windows)
    train_windows_pca = [pca.transform(window) for window in train_windows]
    test_windows, _ = normalize_windows(test_windows, scaler=train_scaler)
    test_windows_pca = [pca.transform(window) for window in test_windows]
    normalized_folds.append((train_windows_pca, train_labels, test_windows_pca, test_labels))
    scalers.append(train_scaler)
    pcas.append(pca)

plot_pca(pcas)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
labels = np.array(labels).reshape(-1, 1)
encoded_labels = encoder.fit_transform(labels)

# Save normalized folds and scalers
joblib.dump(normalized_folds, 'normalized_folds_pca_allData.pkl')
joblib.dump(scalers, 'scalers_pca_allData.pkl')
joblib.dump(encoded_labels, 'encoded_labels_pca_allData.pkl')
joblib.dump(encoder, 'encoder_pca_allData.pkl')
joblib.dump(pcas, 'pcas_pca_allData.pkl')

# Now normalized_folds contains the training and validation sets for each fold

print("Done preparing data")
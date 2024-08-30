import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


sampling_freq = 50
seconds_to_trim = 4

def trim_start(df, lines_to_trim):
    return df.iloc[lines_to_trim*seconds_to_trim:]

def trim_to_peaks(data, plot):
    pressure_data = -(data['pressure']-np.mean(data['pressure']))
    peaks = []
    prom_min = 150

    while (max(pressure_data) > 500):
        new_start = np.argmax(pressure_data)+1
        pressure_data = pressure_data[new_start:].reset_index(drop=True)
        data = data[new_start:].reset_index(drop=True)

    while (len(peaks) < 20):
        prom_min = prom_min - 2
        peaks, _ = find_peaks(pressure_data, prominence=[prom_min, 300])
    
    print(f'Number Peaks: {len(peaks)}')
    
    start_idx = max(peaks[0]-sampling_freq, 0)
    end_idx = min(peaks[-1]+sampling_freq, data.shape[0])

    if plot or (len(peaks)!= 20):
        plt.plot(pressure_data)
        plt.plot(peaks, pressure_data[peaks], "x")
        plt.axvline(x = start_idx, color = 'b')
        plt.axvline(x = end_idx, color = 'b')
        #plt.ylim((-200, 200))
        plt.show()

    return data[start_idx:end_idx]

def getSoftness(filename):
    if (filename.__contains__('dragon')) or (filename.__contains__('Dragon')):
        softness = 'dragonskin30'
    elif (filename.__contains__('flex20')) or (filename.__contains__('Flex20')):
        softness = 'echoflex20'
    elif (filename.__contains__('flex30')) or (filename.__contains__('Flex30')):
        softness = 'echoflex30'
    elif (filename.__contains__('foam')) or (filename.__contains__('Foam')):
        softness = 'foam'
    else:
        print(f"Unknown Softness Value: {filename}")
        raise ValueError
    return softness

def load_csv_files(directories):
    data_frames = []
    required_columns = ["accx", "accy", "accz", "gx", "gy", "gz", "pressure"]
    total_files = 0
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.startswith("EXP") and filename.endswith(".csv"):
                # Extract label from the filename
                label = filename.split('_')[0].replace("EXP", "")
                df = pd.read_csv(os.path.join(directory, filename))
                df = trim_to_peaks(df, False)#, sampling_freq)  # Remove the first 'lines_to_trim' lines
                df = df[required_columns]  # Only keep the required columns
                df['label'] = label  # Add the label column
                df['softness'] = "None"  # Add the softness column
                data_frames.append(df)
                total_files = total_files+1
            if filename.startswith("material_"):
                for fName in os.listdir(directory+"/"+filename):
                    if not fName.endswith("delay100.csv"):
                        if (fName.startswith("M") and fName.endswith(".csv")) or (fName.startswith("EXP") and fName.endswith(".csv")):
                            # Extract label from the filename
                            label = filename.split('_')[1]
                            df = pd.read_csv(os.path.join(directory, filename, fName))
                            df = trim_to_peaks(df, False)#, sampling_freq)  # Remove the first 'lines_to_trim' lines
                            df = df[required_columns]  # Only keep the required columns
                            df['label'] = label  # Add the label column
                            df['softness'] = "None"  # Add the softness column
                            data_frames.append(df)
                            total_files = total_files+1
            if filename.endswith("_just_softness"):
                for fName in os.listdir(directory+"/"+filename):
                    if fName.endswith("delay150.csv"):
                        label = 18 # Add tape as material 18
                        softness = getSoftness(filename)
                        df = pd.read_csv(os.path.join(directory, filename, fName))
                        df = trim_to_peaks(df, False)#, sampling_freq)  # Remove the first 'lines_to_trim' lines
                        df = df[required_columns]  # Only keep the required columns
                        df['label'] = label  # Add the label column
                        df['softness'] = softness  # Add the softness column
                        data_frames.append(df)
                        total_files = total_files+1
            if directory == "softness&texture":
                for fabric in os.listdir(directory+"/"+filename):
                    for fName in os.listdir(directory+"/"+filename+"/"+fabric):
                        if fName.endswith("delay100.csv"):
                            label =  fabric.split('fabric')[1]
                            softness = getSoftness(filename)
                            df = pd.read_csv(os.path.join(directory, filename, fabric, fName))
                            df = trim_to_peaks(df, False)#, sampling_freq)  # Remove the first 'lines_to_trim' lines
                            df = df[required_columns]  # Only keep the required columns
                            df['label'] = label  # Add the label column
                            df['softness'] = softness  # Add the softness column
                            data_frames.append(df)
                            total_files = total_files+1

    print(f"Total files used: {total_files}")
    return pd.concat(data_frames, ignore_index=True)

def create_windows(data, window_size, overlap=0):
    step_size = int(window_size * (1 - overlap))
    windows = []
    labels = []
    for label in data['label'].unique():
        label_data = data[data['label'] == label]
        for i in range(0, len(label_data) - window_size + 1, step_size):
            windows.append(label_data.iloc[i:i + window_size].drop(columns=['label', 'softness']).values)
            labels.append([int(label), label_data.iloc[i].softness])
    return np.array(windows), np.array(labels)

def split_into_folds(windows, labels, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in kf.split(windows):
        folds.append((windows[train_index], labels[train_index], windows[test_index], labels[test_index]))
        a, b = np.unique(labels[train_index, 0], return_counts=True)
        print(b)
        a, b = np.unique(labels[train_index, 1], return_counts=True)
        print(b)
        a, b = np.unique(labels[test_index, 0], return_counts=True)
        print(b)
        a, b = np.unique(labels[test_index, 1], return_counts=True)
        print(b)
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
directory = ['collected_data_previous', 'collected_data', 'softness_data', 'softness&texture']
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
softness_encoder = OneHotEncoder(sparse=False)
softness = np.array(labels[:, 1]).reshape(-1, 1)
encoded_softness = softness_encoder.fit_transform(softness)

labels_encoder = OneHotEncoder(sparse=False)
labels = np.array(labels[:, 0].astype(np.int)).reshape(-1, 1)
encoded_labels = labels_encoder.fit_transform(labels)

# Save normalized folds and scalers
joblib.dump(normalized_folds, 'normalized_folds_pca_allData_peakTrim_softness.pkl')
joblib.dump(scalers, 'scalers_pca_allData_peakTrim_softness.pkl')
joblib.dump(encoded_labels, 'encoded_labels_pca_allData_peakTrim_softness.pkl')
joblib.dump(labels_encoder, 'labelsencoder_pca_allData_peakTrim_softness.pkl')
joblib.dump(encoded_softness, 'encoded_softness_pca_allData_peakTrim_softness.pkl')
joblib.dump(softness_encoder, 'softnessencoder_pca_allData_peakTrim_softness.pkl')
joblib.dump(pcas, 'pcas_pca_allData_peakTrim_softness.pkl')

# Now normalized_folds contains the training and validation sets for each fold

print("Done preparing data")
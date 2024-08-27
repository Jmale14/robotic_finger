import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy, pylab
import array
from scipy.signal import butter, lfilter, resample
from scipy import signal
import os
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

# open the directory that you want to use data
current_dir = os.getcwd()
new_dir = os.chdir(r'F:\robotic_finger\collected_data\material_14')


file_list=['M14_EXP1.csv','M14_EXP2.csv','M14_EXP3.csv'] 

#define dictionary
main_dataframe = {'texture1':pd.DataFrame(pd.read_csv(file_list[0])), 'texture2': pd.DataFrame(pd.read_csv(file_list[1])),
                  'texture3': pd.DataFrame(pd.read_csv(file_list[2]))} 


# #create a file list with the name of collected data
# file_list=['EXP1.csv','EXP1_r20_s90_dist5_delay100.csv','EXP2_r20_s90_dist5_delay100.csv', 
#            'EXP3_r20_s90_dist5_delay100.csv', 'EXP4_r20_s90_dist5_delay100.csv', 'EXP5_r20_s90_dist5_delay100.csv'] 

# #define dictionary
# main_dataframe = {'texture1':pd.DataFrame(pd.read_csv(file_list[0])), 'texture2': pd.DataFrame(pd.read_csv(file_list[1])),
#                   'texture3': pd.DataFrame(pd.read_csv(file_list[2])), 'texture4': pd.DataFrame(pd.read_csv(file_list[3])),
#                   'texture5': pd.DataFrame(pd.read_csv(file_list[4])), 'texture6': pd.DataFrame(pd.read_csv(file_list[5]))} 

#val = list(main_dataframe)[1] #get the name of the first member of the dict
#first_val = list(main_dataframe.values())[0] #get the values from the first dataframe


common_size = 1180

for key in main_dataframe.keys():
    main_dataframe[key] = main_dataframe[key].head(common_size) #trim to common size rows
    
#remove the first 295 data points because the first move has different frequency    
for key in main_dataframe.keys():
    main_dataframe[key] = main_dataframe[key].iloc[294:]
    
#%%% Plot the chosen signal

#select the dataset that you want to plot
dataset_val = 0

plt.figure(dpi=300)

NameTitle = 'Pressure Output'


# plt.plot(list(main_dataframe.values())[dataset_val].iloc[0:49]['time'],
#          list(main_dataframe.values())[dataset_val].iloc[0:49]['pressure'],color='black')

plt.plot(list(main_dataframe.values())[dataset_val].iloc[0:886]['time'],
         list(main_dataframe.values())[dataset_val].iloc[0:886]['accx'],color='black')

plt.title(NameTitle)
plt.legend(['Pressure'], bbox_to_anchor=(1.24, 1.03),loc="upper right")
plt.xlabel("Time (s)")

        
 #  plt.savefig(saveName + '_low.png',bbox_inches="tight") #use bbox_inches to save the whole image with the legend

#%%% Normalize the selected features and plot

# def NormalizeData(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

#z-score Normalization
def NormalizeData(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev


normalized_dict = {}

for k, df in main_dataframe.items():
    # Selecting the relevant columns from the dataframe
    temp = df.iloc[:, [2, 3, 4, 5, 6, 7, 10]]
    
    # Initialize an empty DataFrame to store normalized rows
    normalized_temp = pd.DataFrame(columns=temp.columns, index=temp.index)
    
    # Apply normalization row-wise
    for index, row in temp.iterrows():
        normalized_temp.loc[index] = NormalizeData(row.values)
    
    # If you need to maintain the original dataframe structure,
    # you may want to integrate the normalized data back into the original dataframe.
    # This example just updates the specified columns with their normalized values.
    for col in temp.columns:
        df[col] = normalized_temp[col]
    
    normalized_dict[k] = df

main_dataframe_normalized = {'texture1':normalized_dict['texture1'], 'texture2':normalized_dict['texture2'],
                             'texture3':normalized_dict['texture3'],'texture4':normalized_dict['texture4'],
                             'texture5':normalized_dict['texture5'], 'texture6':normalized_dict['texture6']}

plt.figure(dpi=300)
plt.plot(list(main_dataframe_normalized.values())[3])
plt.show()    

#%%% Plot accelerometer, gyroscope

def plot_dataframes_together(dataframes,data_points):
    """
    Plot data from multiple dataframes together using scatter plot.
    
    Parameters:
        dataframes (list): List of dataframes to plot.
    """
    colors = ['b', 'r', 'g', 'c', 'm', 'y' ]  # Colors for different dataframes
    plt.figure(dpi=300)
    
    for i, df in enumerate(dataframes):
        x_axis = df.iloc[0:data_points]['gz']
        y_axis = df.iloc[0:data_points]['accz']
        plt.scatter(x_axis, y_axis, label=f'Dataframe {i}', color=colors[i])

    # for i, df in enumerate(dataframes):
    #    # plt.axes(projection ="3d")
    #     x_axis = df.iloc[0:data_points]['gz']
    #     y_axis = df.iloc[0:data_points]['accz']
    #     z_axis = df.iloc[0:data_points]['pressure']
        
    #     plt.scatter(x_axis, y_axis, z_axis, label=f'Dataframe {i}', color=colors[i])
        
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data from Multiple Dataframes')
    plt.legend()
    plt.show()

# Example usage:
dataframes = [main_dataframe_normalized['texture1'], main_dataframe_normalized['texture2'], main_dataframe_normalized['texture3'],
              main_dataframe_normalized['texture4'], main_dataframe_normalized['texture5'], main_dataframe_normalized['texture6']]  # Replace df1, df2, df3 with your actual dataframes
plot_dataframes_together(dataframes,192)



#%%%

# Function to extract signals and create input readings
def create_input_readings(df):
    #accelerometer_data = df.iloc[:, 2:5].values  # Columns 3 to 5 are accelerometer signals
    #gyroscope_data = df.iloc[:, 5:8].values      # Columns 6 to 8 are gyroscope signals
    #angle_data = df.iloc[:, 8:10].values      # Columns 6 to 8 are gyroscope signals
    accelerometer_data = df.iloc[:, 8].values  # Columns 3 to 5 are accelerometer signals
    gyroscope_data = df.iloc[:, 9].values      # Columns 6 to 8 are gyroscope signals    
    pressure_data = df.iloc[:, 10].values        # Column 11 is pressure signal
    
    input_readings = []
    signal_size = 24 #previous number is 49
    
    for i in range(0, len(df) - (signal_size-1), signal_size):  # Create sliding windows
         input_reading = np.concatenate((np.expand_dims(accelerometer_data[i:i+signal_size],axis=1),
                                         np.expand_dims(gyroscope_data[i:i+signal_size],axis=1),
                                         np.expand_dims(pressure_data[i:i+signal_size], axis=1)), axis=1)
        
        #input_reading = np.concatenate((accelerometer_data[i:i+signal_size, :],
        #                                 gyroscope_data[i:i+signal_size, :],
        #                                 angle_data[i:i+signal_size, :],
        #                                 np.expand_dims(pressure_data[i:i+signal_size], axis=1)), axis=1)
                                                 
         input_readings.append(input_reading)
    
    return input_readings

# Create input samples and labels
X = []
y = []

for texture, dataframe in main_dataframe_normalized.items():
    input_readings = create_input_readings(dataframe)
    X.extend(input_readings)
    y.extend([texture] * len(input_readings))
            
# Replace 'texture1' with 0,'texture2' with 1,'texture3' with 2,
y = [0 if x == 'texture1' else x for x in y]
y = [1 if x == 'texture2' else x for x in y]
y = [2 if x == 'texture3' else x for x in y]    
y = [3 if x == 'texture4' else x for x in y]    
y = [4 if x == 'texture5' else x for x in y]    
y = [5 if x == 'texture6' else x for x in y]   
     
# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

#Convert data type from object to float. This is required for Tensorflow
X = X.astype(float)
    
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

# Verify the shapes of input samples and labels
print("Shape of input samples:", X.shape)
print("Shape of labels:", y.shape)


#%%% Plot input reading from the previous section

#select the dataset that you want to plot
select_sample = 12;
#select the column, 6 is pressure
select_column = 2
#Adjust the quality of the plot
plt.figure(dpi=300)

NameTitle = ''

# plt.plot(list(main_dataframe.values())[dataset_val].iloc[0:49]['time'],
#          list(main_dataframe.values())[dataset_val].iloc[0:49]['pressure'],color='black')

plt.plot(X_train[select_sample,:,select_column],color='black')
#plt.plot(X_train[select_sample],color='black')

plt.title(NameTitle)
plt.legend([''], bbox_to_anchor=(1.24, 1.03),loc="upper right")
plt.xlabel("Time (s)")



#%%%

import numpy as np

# Assuming your accelerometer, gyroscope, and pressure data are numpy arrays
accelerometer_data = np.random.rand(886, 3)
gyroscope_data = np.random.rand(886, 3)
pressure_data = np.random.rand(886, 1)

# Define the size of the input reading
window_size = 24 #49 previous window size
stride = (accelerometer_data.shape[0] - window_size) // 8 # Divide by 8 to get 9 input readings

# Initialize lists to store grouped data
grouped_data = []

# Generate 9 input readings
for i in range(9):
    start_index = i * stride
    end_index = start_index + window_size
    
    # Extract data for the current input reading
    accelerometer_reading = accelerometer_data[start_index:end_index, :]
    gyroscope_reading = gyroscope_data[start_index:end_index, :]
    pressure_reading = pressure_data[start_index:end_index, :]
    
    # Combine the data into one input reading
    input_reading = np.concatenate((accelerometer_reading, gyroscope_reading, pressure_reading), axis=1)
    
    # Append the input reading to the list of grouped data
    grouped_data.append(input_reading)

# Convert the list of grouped data into a numpy array
grouped_data = np.array(grouped_data)

# Print the shape of the grouped data
print(grouped_data.shape)

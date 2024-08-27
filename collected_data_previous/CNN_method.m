%%

% Define file paths
filePaths = {'EXP3_r20_s90_dist5_delay100.csv', 'EXP4_r20_s90_dist5_delay100.csv', ...
             'EXP5_r20_s90_dist5_delay100.csv', 'EXP6_r20_s90_dist5_delay100.csv', ...
             'EXP7_r20_s90_dist5_delay100.csv', 'EXP8_r20_s90_dist5_delay100.csv', ...
             'EXP9_r20_s90_dist5_delay100.csv'};

allData = [];
allLabels = [];

for i = 1:length(filePaths)
    data = readtable(filePaths{i});
    data = data(296:end, :);  % Remove the first 295 data points
    data = data(1:882, :);  % Ensure each dataset is only 882 datapoints

    features = data{:, {'gz', 'pressure'}};
    numSamples = floor(size(features, 1) / 49);  % Ensures integer number of samples
    reshapedFeatures = reshape(features', [2, 49, 1, numSamples]);  % Reshape for CNN input

    labels = repmat(i, numSamples, 1);
    allData = cat(4, allData, reshapedFeatures);
    allLabels = [allLabels; labels];
end

% Convert labels to categorical
allLabels = categorical(allLabels);

% Randomly split data into training and testing sets
idx = randperm(size(allData, 4));
numTrain = floor(0.8 * length(idx));
trainIdx = idx(1:numTrain);
testIdx = idx(numTrain+1:end);

trainData = allData(:,:,:,trainIdx);
trainLabels = allLabels(trainIdx);
testData = allData(:,:,:,testIdx);
testLabels = allLabels(testIdx);

layers = [
    imageInputLayer([2 49 1])
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
    fullyConnectedLayer(320) %280
    % Add an ReLU non-linearity.
    reluLayer
    fullyConnectedLayer(numel(unique(allLabels)))
    softmaxLayer
    classificationLayer
];

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.0009, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.006, ...
    'MaxEpochs', 36, ...
    'MiniBatchSize', 4, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testData, testLabels});

net = trainNetwork(trainData, trainLabels, layers, opts);
%%
% Prediction and accuracy calculation
predictedLabels = classify(net, testData);
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy * 100, '%.2f'), '%']);

% Plotting the confusion matrix
figure;
confusionchart(testLabels, predictedLabels);
title('Confusion Matrix');

% % Optional: Plot normalized confusion matrix
% figure;
% cm = confusionchart(testLabels, predictedLabels, 'Normalization', 'total');
% cm.Title = 'Normalized Confusion Matrix';
% cm.RowSummary = 'row-normalized';
% cm.ColumnSummary = 'column-normalized';

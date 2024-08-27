params = load("C:\Users\jjm53\OneDrive - University of Bath\robotic_finger\collected_data_previous\params_2024_06_17__17_57_07.mat");

layers = [
    sequenceInputLayer(7,"Name","sequence","MinLength",90,"Normalization","zscore","NormalizationDimension","channel")
    lstmLayer(128,"Name","lstm1")
    reluLayer()
    lstmLayer(128,"Name","lstm2",OutputMode="last")
    reluLayer()
    fullyConnectedLayer(100,"Name","fc1")
    reluLayer()
    fullyConnectedLayer(100,"Name","fc2")
    reluLayer()
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(experiments,"Name","fcend")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput", "Classes", labels)];

%plot(layerGraph(layers));

numObservations = numel(windowedData.X);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);
XTrain = windowedData.X(idxTrain);
TTrain = categorical(windowedData.Y(idxTrain)', [0:experiments-1]);

XVal = windowedData.X(idxValidation);
TVal = categorical(windowedData.Y(idxValidation)', [0:experiments-1]);

XTest = windowedData.X(idxTest);
TTest = categorical(windowedData.Y(idxTest)', [0:experiments-1]);

% numObservations = numel(XTrain);
% for i=1:numObservations
%     sequence = XTrain{i};
%     sequenceLengths(i) = size(sequence,1);
% end
% 
% [sequenceLengths,idx] = sort(sequenceLengths);
% XTrain = XTrain(idx);
% TTrain = TTrain(idx);

% figure
% bar(sequenceLengths)
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

options = trainingOptions("adam", ...
    MaxEpochs=250, ...
    InitialLearnRate=0.002,...
    GradientThreshold=1, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
Verbose=false, ...
BatchNormalizationStatistics="population", ...
ValidationData = {XVal, TVal}, ...
ValidationFrequency=10, ...
OutputNetwork="best-validation-loss");    
%Metrics="accuracy", ...
    %Verbose=false);

net = trainNetwork(XTrain,TTrain,layers,options);



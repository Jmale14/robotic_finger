experiments = 18;
win_len = 2; %sec
freq = 45; %Hz
length_2_cut = 1; %sec
windowedData = struct;
windowedData.X = {};%cell(1);
windowedData.Y = [];%cell(1);

labels = categorical([0:experiments-1]);

for i = 0:experiments-1
    fprintf("Loading data %i of %i \n", i+1, experiments)
    newData = importfile("EXP"+i+"_r20_s90_dist5_delay100.csv", [2, Inf]);
    windowedData = windowData(newData, windowedData, win_len*freq, length_2_cut*freq, i);
end
%windowedData.X(1) = [];
%windowedData.Y(1) = [];

%inputs = windowedData(2, :);
%outputs = windowedData(1, :);

%inputs = squeeze(cell2mat(cellfun(@(x)reshape(x,[1,1,size(x)]),inputs,'un',0)));
%%inputs = reshape(inputs, 216, []);
%outputs = cell2mat(outputs);

%windowedData(2, :) = string2cell(string(windowedData(2, :)));
%ds = arrayDatastore(windowedData', "OutputType","same", "IterationDimension", 1, "ReadSize", 1);


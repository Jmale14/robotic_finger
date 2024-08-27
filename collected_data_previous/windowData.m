function [windowedData] = windowData(newData, windowedData, win_len_samples, cut_len_samples, class)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

start_idx = cut_len_samples;
end_idx = start_idx + win_len_samples-1;

i=width(windowedData.X)+1;
while end_idx <= height(newData)
    windowedData.Y(i) = class;%class;%repmat(categorical(class), 1, win_len_samples);
    windowedData.X{i} = newData{start_idx:end_idx, {'accx', 'accy', 'accz', 'gx', 'gy', 'gz', 'pressure'}}';
    i = i+1;
    start_idx = start_idx+win_len_samples;
    end_idx = start_idx + win_len_samples-1;
end

end
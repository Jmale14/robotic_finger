

YTest = classify(net, XTest);
%mode_YTest = {};
%mode_TTest = {};
% for i = 1:length(YTest)
%     mode_YTest{i} = mode(YTest{i});
%     mode_TTest{i} = mode(TTest{i});
% end

%mode_YTest = reshape([mode_YTest{:}],1,[]);
%mode_TTest = reshape([mode_TTest{:}],1,[]);

acc = mean(YTest == TTest)



figure
c = confusionchart(TTest, YTest);
sortClasses(c, labels)
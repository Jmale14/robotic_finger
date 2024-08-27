% Get a list of all files in the folder
contentsList = dir();
datasets = {};
dataset_count = 0;
for p = 3:length(contentsList)
    k = contentsList(p);
    if and(k.isdir, startsWith(k.name, "hp_search_"))
        dataset_count = dataset_count+1;
        fileList = dir(k.name);
        % Iterate over the list of files
        for i = 1:length(fileList)
            % Get the name of the current file
            fileName = fileList(i).name;
            
            % Check if the file name starts with the specified pattern
            if startsWith(fileName, 'train_hist_f1_score')
                % Display the file name
                fprintf('File found: %s\n', fileName);
                dataset = load_results_file([k.name '/' fileName]);
                datasets{1,dataset_count} = dataset;
            elseif startsWith(fileName, 'train_hist_accuracy')
                % Display the file name
                fprintf('File found: %s\n', fileName);
                dataset = load_results_file([k.name '/' fileName]);
                datasets{2,dataset_count} = dataset;
            elseif startsWith(fileName, 'trial_details_')
                % Display the file name
                fprintf('File found: %s\n', fileName);
                trialDetails = importTrialDetails([k.name '/' fileName]);
                datasets{3,dataset_count} = trialDetails;
            end
        end
    end
end


%% Plot F1 against hyperparameter
bestTrial = [0,0,0];
for hp = string(trialDetails.Properties.VariableNames)
    if and(hp ~= "trial", hp ~= "HP_EPOCHS")
        figure
        hold on
        for hpSearch = 1:width(datasets)
            trialDetails = datasets{3, hpSearch};
            dataset = datasets{1, hpSearch};
            dataset_acc = datasets{2, hpSearch};
            for trial = table2array(trialDetails(:,'trial'))'
                [maxF1, i] = max(table2array(dataset(:, "val_average_trial"+trial)));
                if maxF1 > bestTrial(1)
                    bestTrial = [maxF1, hpSearch, trial, table2array(dataset_acc(i, "val_average_trial"+trial))];
                end
                scatter(table2array(trialDetails(trialDetails.trial==trial, hp)), maxF1)
            end
        end
        title(hp)
        hold off
    end
end

%% Find best trial
figure
hold on
% yyaxis left
plot(table2array(datasets{1, bestTrial(2)}(:, "val_average_trial"+bestTrial(3))),'DisplayName','Val Average')
plot(movavg(table2array(datasets{1, bestTrial(2)}(:, "val_average_trial"+bestTrial(3))),'simple',10),'DisplayName','Val Average MA')
plot(table2array(datasets{1, bestTrial(2)}(:, "average_trial"+bestTrial(3))),'DisplayName','Train Average')
% yyaxis right
plot(table2array(datasets{1, bestTrial(2)}(:, "val_std_trial"+bestTrial(3))),'DisplayName','Val Std')
legend()
title("F1 Score")
hold off

figure
hold on
% yyaxis left
plot(table2array(datasets{2, bestTrial(2)}(:, "val_accuracy_1_trial"+bestTrial(3))),'DisplayName','1')
plot(table2array(datasets{2, bestTrial(2)}(:, "val_accuracy_2_trial"+bestTrial(3))),'DisplayName','2')
plot(table2array(datasets{2, bestTrial(2)}(:, "val_accuracy_3_trial"+bestTrial(3))),'DisplayName','3')
plot(table2array(datasets{2, bestTrial(2)}(:, "val_accuracy_4_trial"+bestTrial(3))),'DisplayName','4')
plot(table2array(datasets{2, bestTrial(2)}(:, "val_accuracy_5_trial"+bestTrial(3))),'DisplayName','5')

plot(table2array(datasets{2, bestTrial(2)}(:, "val_average_trial"+bestTrial(3))),'DisplayName','Val Average')
plot(movavg(table2array(datasets{2, bestTrial(2)}(:, "val_average_trial"+bestTrial(3))),'simple',10),'DisplayName','Val Average MA')
plot(table2array(datasets{2, bestTrial(2)}(:, "average_trial"+bestTrial(3))),'DisplayName','Train Average')
% yyaxis right
plot(table2array(datasets{2, bestTrial(2)}(:, "val_std_trial"+bestTrial(3))),'DisplayName','Val Std')
legend()
title("Accuracy")
hold off

fprintf("Best Trial Details:")
datasets{3, bestTrial(2)}(datasets{3, bestTrial(2)}.trial==bestTrial(3),:)
disp(bestTrial(1))
disp(bestTrial(4))







% This script finds the optimum TSK model for a high dimensioning dataset, by using the grid search method.
% Then the optimum TSK model is training.

% Author: Tzouvaras Evangelos
% Email: tzouevan@ece.auth.gr

%% Clear variables and command window
clear;
clc;

%% Load the data from .data file and sort them based on indices
data=csvread('epileptic_seizure_data.csv',1,1);

%% Split and normalize the data into three subsets by using the split scale function
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

%% Initialize the possible values for parameters
R_values = [0.2 ,0.3, 0.4, 0.5, 0.6]; 
features_number = [5, 10, 15, 20, 25];       

%% Local-Helpful variables
Num_of_folds = 5;                                   % The number of created folds on the training data
mean_errors = zeros(1,(length(features_number)*length(R_values)));    % An array to save each mean error 
count = 1;
fold_mean_erros = zeros(1,Num_of_folds);            % Create an array to save the mean errors for each fold 

%% Grid Search method
% 5-fold cross validation
cv = cvpartition(size(trnData,1), 'KFold', Num_of_folds); 
% Relief algorithm: Select specific features_number data from 83 total features
[Index,~] = relieff(trnData(:,1:end-1),trnData(:,end),10);

% 2 stage for loop for all the combinations of parameters' values - Grid
% search method
for i = 1:length(features_number)                       % For each features_number value
    for j = 1:length(R_values)                          % For each R_values
        % 5 folds to calculate a mean error for each model
        for k = 1:Num_of_folds                          % For each fold
            % Take the training and the validation data for each fold, by
            % using the created indexes from relieff algorithm
            fold_trnData = trnData(cv.training(k) == 1, [Index(1:features_number(i)), end]);
            fold_chkData = trnData(cv.test(k) == 1, [Index(1:features_number(i)), end]);
            
            % Create the fis by using the fold data
            Options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', R_values(j));
            fis = genfis(fold_trnData(:,1:(end-1)), fold_trnData(:,end), Options);

            % Train the fis and calculate the error
            [trnFis,trnError,~,valFis,valError]=anfis(fold_trnData,fis,[30 0 0.01 0.9 1.1],[],fold_chkData);  
    
            % Calculate the mean error for each fold
            fold_mean_erros(k) = min(valError);
        end
        % Calculate and save the mean error for this set of parameters
        mean_errors(count) = mean(fold_mean_erros); 
        count = count + 1;
    end
end

%% Plot the 3-D graph between mean_errors, features_number and R_values
% Create a grid of (r_a, feature_numbers) combinations
[R, Features] = meshgrid(R_values, features_number);

% Reshape the error array to match the grid
Errors_Reshaped = reshape(mean_errors, numel(R_values), numel(features_number));

% Create a 3D scatter plot
figure;
scatter3(R(:), Features(:), Errors_Reshaped(:), 'filled');
xlabel('Radius of clusters(r_a)');
ylabel('Number of Features');
zlabel('Mean Error');
title('Relationship between Radius, Number of Features and Mean Error');

%% Find the optimum TSK model
[min_error, min_error_index] = min(Errors_Reshaped(:));
[optimum_r_a_index, optimum_feature_number_index] = ind2sub(size(Errors_Reshaped), min_error_index);
optimum_radius = R_values(optimum_r_a_index);
optimum_feature_number = features_number(optimum_feature_number_index);

%% Train the optimum TSK model
% Create the datasets by using the Indexes fro, the relieff algorithm
optimum_trnData = trnData(:, [Index(1:optimum_feature_number), end]);
optimum_chkData = chkData(:, [Index(1:optimum_feature_number), end]);
            
% Create the fis by using the above data
Options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', optimum_radius);
optimum_fis = genfis(optimum_trnData(:,1:(end-1)), optimum_trnData(:,end), Options);

% Train the fis and calculate the error
[trnFis,trnError,~,valFis,valError]=anfis(optimum_trnData,optimum_fis,[100 0 0.01 0.9 1.1],[],optimum_chkData);  

%% Plot the prediction error for the optimum model 
% Calculate the output of the validation fis and by using the chkData subset
Y = round(evalfis(valFis, chkData(:,Index(1:optimum_feature_number))));    
    
% Calculate the prediction error
optimum_error = chkData(:,end) - Y;
    
% Create the prediction error plot
figure; 
plot(optimum_error);
title('Optimum TSK model: Prediction Error');
ylabel('Error');
xlabel('Sample index');

%% Plot the learning curves of the optimum model
figure;
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); 
ylabel('Error');
legend('Training Error','Validation Error');
title('Optimum TSK model: Learning Curves');

%% Plot the fuzzy sets before training for the optimum model
figure;
sgtitle(sprintf('TSK membership functions before training'));
% Helpful variables for the subplot
a_1 = round(sqrt(optimum_feature_number));
a_2 = round(optimum_feature_number/a_1) + 1;

for i = 1:optimum_feature_number
    [x,mf] = plotmf(optimum_fis,'input',i);
    subplot(a_1,a_2,i);
    plot(x,mf);
    xlabel(sprintf('Membership Functions for Input %d',i));
end

%% Plot the fuzzy sets after training for the optimum model
figure;
sgtitle(sprintf('TSK membership functions after training'));

for i = 1:optimum_feature_number
    [x,mf] = plotmf(trnFis,'input',i);
    subplot(a_1,a_2,i);
    plot(x,mf);
    xlabel(sprintf('Membership Functions for Input %d',i));
end

%% Calculate the evaluation parameters for each TSK model   
% Error matrix
classes = unique(data(:, end));
errorMatrix = zeros(length(classes));
   
for i = 1:length(tstData)
    x = find(classes == Y(i));
    y = find(classes == data(i, end));
    errorMatrix(x, y) = errorMatrix(x, y) + 1;
end
   
% Overall Accuracy 
OA = trace(errorMatrix)/length(tstData);
    
% Producer's Accuracy
x_jc = sum(errorMatrix);
    
for k = 1:length(x_jc)
   PA(k) =  errorMatrix(k,k)/x_jc(k);
end
    
% User's Accuracy
x_ir = sum(errorMatrix,2);
   
for k = 1:length(x_ir)
   UA(k) =  errorMatrix(k,k)/x_ir(k);
end

% K parameter
sum_of_prods = 0;
for k = 1:length(PA)
   sum_of_prods = sum_of_prods + PA(k)*UA(k); 
end
k = (length(tstData) * trace(errorMatrix) - sum_of_prods) / (length(tstData)^2 - sum_of_prods);
    
%% Print the evaluations parameters for each TSK model
fprintf('\n=========================================================\n');
fprintf('Optimum TSK model');
% Print the Error Matrix that is calculated for each TSK model 
fprintf('\n\nError Matrix:\n');
fprintf('%d\t%d\t%d\t%d\t%d\t\n', errorMatrix);
    
% Print the OA, UA, K parameters
fprintf('\nOverall Accuracy(OA): %.2f', OA);
fprintf('\nProducer Accuracy(PA): %.2f', PA);
fprintf('\nUser Accuracy(UA): %.2f', UA);
fprintf('\nK parameter: %.2f', k);
    
fprintf('\n=========================================================\n');


% This script is for training two TSK models  by using the "Haberman's Survival" dataset with Subtractive Clustering method(TSK Classification problem)

% Author: Tzouvaras Evangelos
% Email: tzouevan@ece.auth.gr

%% Clear variables and command window
clear;
clc;

%% Load the data from .data file and sort them based on indices
data=load('haberman.data');

% Sort the dataset based on the values in the final column
sorted_data = sortrows(data, 4);

%% Split and normalize the sorted data into three subsets by using the split scale function
preproc=1;
[trnData,chkData,tstData]=split_scale(sorted_data,preproc);

%% Set up the training models 1,2 that uses the class independent method
% For the first TSK model - Class Independent - Small Radius = 0.2 
Options(1) = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.2);
% For the second TSK model - Class Independent - Large Radius = 0.8
Options(2) = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.8);

%% A for loop that runs for each TSK model
for i=1:2
    %% Create the fuzzy inference system (fis)
    fis = genfis(trnData(:,1:(end-1)), trnData(:,end), Options(i));
    
    %% Training 
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);
    
    %% Evaluate the TSK models
    Y = round(evalfis(tstData(:,1:end-1),valFis));
       
    %% Plot the fuzzy sets after training
    figure;
    sgtitle(sprintf('TSK membership functions after training \nNumber of TSK model: %d', i));
       
    [x,mf] = plotmf(trnFis,'input',1);
    subplot(3,3,1);
    plot(x,mf);
    xlabel('Membership Functions for Input 1');
        
    [x,mf] = plotmf(trnFis,'input',2);
    subplot(3,3,2);
    plot(x,mf);
    xlabel('Membership Functions for Input 2');
        
    [x,mf] = plotmf(trnFis,'input',3);
    subplot(3,3,3);
    plot(x,mf);
    xlabel('Membership Functions for Input 3');
       
    %% Plot the learning curves - Training error and Validation error 
    figure;
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); 
    ylabel('Error');
    legend('Training Error','Validation Error');
    title('TSK model 1: Learning Curves');
    
    %% Calculate the evaluation parameters for each TSK model   
    % Error matrix
    classes = unique(data(:, end));
    errorMatrix = zeros(length(classes));
    
    for j = 1:length(tstData)
        x = find(classes == Y(j));
        y = find(classes == data(j, end));
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
    fprintf('TSK model %d', i);
    % Print the Error Matrix that is calculated for each TSK model 
    fprintf('\n\nError Matrix:\n');
    fprintf('%d\t%d\t\n', errorMatrix);
    
    % Print the OA, UA, K parameters
    fprintf('\nOverall Accuracy(OA): %.2f', OA);
    fprintf('\nProducer Accuracy(PA): %.2f', PA);
    fprintf('\nUser Accuracy(UA): %.2f', UA);
    fprintf('\nK parameter: %.2f', k);
    
    fprintf('\n=========================================================\n');
    
%% End of for loop
end


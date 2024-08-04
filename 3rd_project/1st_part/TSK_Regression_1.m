% This script is for training four TSK models  by using the "Airfoil Self-Noise" dataset with grid partitioning method(TSK Regression problem)

% Author: Tzouvaras Evangelos
% Email: tzouevan@ece.auth.gr

%% Load the data from .dat file
data=load('airfoil_self_noise.dat');

%% Split and normalize the data into three subsets by using the split scale function
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

%% Set up the training models and their parameters
% For the first TSK model
Options(1) = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, 'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'constant');
% For the second TSK model
Options(2) = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, 'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'constant');
% For the third TSK model
Options(3) = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, 'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'linear');
% Fot the forth TSK model  
Options(4) = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, 'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'linear');

%% Extenrnal Loop to repeat the whole process for all FIS
for i = 1:4             
    
    %% Create each fuzzy inference system (fis)
    fis = genfis(trnData(:,1:5), trnData(:,6), Options(i));

    %% Plot the initial fuzzy sets for each ceated fis(before training)
    figure;
    sgtitle(sprintf('TSK membership functions before training \nNumber of TSK model: %d', i));
   
    [x,mf] = plotmf(fis,'input',1);
    subplot(5,3,1);
    plot(x,mf);
    xlabel('Membership Functions for Input 1');
    
    [x,mf] = plotmf(fis,'input',2);
    subplot(5,3,2);
    plot(x,mf);
    xlabel('Membership Functions for Input 2');
    
    [x,mf] = plotmf(fis,'input',3);
    subplot(5,3,3);
    plot(x,mf);
    xlabel('Membership Functions for Input 3');
    
    [x,mf] = plotmf(fis,'input',4);
    subplot(5,3,4);
    plot(x,mf);
    xlabel('Membership Functions for Input 4');
    
    [x,mf] = plotmf(fis,'input',5);
    subplot(5,3,5);
    plot(x,mf);
    xlabel('Membership Functions for Input 5');
 
    %% Training 
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);

    %% Plot the fuzzy sets after training
    figure;
    sgtitle(sprintf('TSK membership functions after training \nNumber of TSK model: %d', i));
   
    [x,mf] = plotmf(trnFis,'input',1);
    subplot(5,3,1);
    plot(x,mf);
    xlabel('Membership Functions for Input 1');
    
    [x,mf] = plotmf(trnFis,'input',2);
    subplot(5,3,2);
    plot(x,mf);
    xlabel('Membership Functions for Input 2');
    
    [x,mf] = plotmf(trnFis,'input',3);
    subplot(5,3,3);
    plot(x,mf);
    xlabel('Membership Functions for Input 3');
    
    [x,mf] = plotmf(trnFis,'input',4);
    subplot(5,3,4);
    plot(x,mf);
    xlabel('Membership Functions for Input 4');
    
    [x,mf] = plotmf(trnFis,'input',5);
    subplot(5,3,5);
    plot(x,mf);
    xlabel('Membership Functions for Input 5');

    %% Plot the learning curves - Training error and Validation error  
    figure;
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); 
    ylabel('Error');
    legend('Training Error','Validation Error');
    title(sprintf('%d TSK model: Learning Curves', i));

    %% Plot the prediction error
    % Calculate the output of the validation fis and by using the chkData
    % subset
    Y = evalfis(valFis, chkData(:,1:5));    
    
    % Calculate the prediction error
    prediction_error = chkData(:,6) - Y;
    
    % Create the prediction error plot
    figure; 
    plot(prediction_error);
    title(sprintf('%d TSK model: Prediction Error', i));
    ylabel('Error');
    xlabel('Sample index');
    
    %% Calculate the evaluation parameters 
    %  Mean Square Error (MSE)
    MSE = mean(prediction_error.^2);
    
    % Root Mean Square Error (RMSE)
    RMSE = sqrt(MSE);
    
    % Coefficient of determination factor (R^2)
    Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);   % Evaluation Function
    R2=Rsq(Y,chkData(:,end));
    
    % Normalized Mean Squared Error (NMSE)
    NMSE = 1 - R2;
    
    % NDEI facotr
    NDEI = sqrt(NMSE);

    % Print all the factors for each training
    fprintf('\n==================================================================================\n');
    fprintf('TSK Model %d: RMSE = %f  NMSE = %f  NDEI = %f  R2 = %f\n', i, RMSE, NMSE, NDEI, R2);
    fprintf('==================================================================================\n');
%% End of for loop
end


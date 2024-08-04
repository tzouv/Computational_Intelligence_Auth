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
[trnData,chkData,tstData]=split_scale(data,preproc);

% Create the TSK models 3 and 4
radius = [0.2 0.8];
% On each iteration create a TSK model
for l=1:2
    [c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius(l));
    [c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius(l));
    % The number of rules
    num_rules=size(c1,1)+size(c2,1);

    % Build FIS From Scratch
    fis=newfis('TSK_FIS','sugeno');

    % Add Input-Output Variables
    names_in={'in1','in2','in3'};
    for i=1:size(trnData,2)-1
        fis=addvar(fis,'input',names_in{i},[0 1]);
    end
    fis=addvar(fis,'output','out1',[1 2]);

    %Add Input Membership Functions
    for i=1:size(trnData,2)-1
        for j=1:size(c1,1)
            name = sprintf('sth%d', j);
            fis=addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
        end
        for j=1:size(c2,1)
             name = sprintf('sth2%d', j);
            fis=addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
        end
    end

    %Add Output Membership Functions
    params=[ones(1,size(c1,1)) 2*ones(1,size(c2,1))];%[zeros(1,size(c1,1)) ones(1,size(c2,1))];
    for i=1:num_rules
        name = sprintf('out%d',i);
        fis=addmf(fis,'output',1,name,'constant',params(i));
    end

    %Add FIS Rule Base
    ruleList=zeros(num_rules,size(trnData,2));
    for i=1:size(ruleList,1)
        ruleList(i,:)=i;
    end
    ruleList=[ruleList ones(num_rules,2)];
    fis=addrule(fis,ruleList);
    
    %% Training 
    [trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);

    %% Evaluate the TSK models
    Y = round(evalfis(tstData(:,1:end-1),valFis));

    %% Plot the fuzzy sets after training
    figure;
    sgtitle(sprintf('TSK membership functions after training \nNumber of TSK model: %d', (l+2)));
       
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
    fprintf('TSK model %d', (l+2));
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

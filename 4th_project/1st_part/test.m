%% Clear variables and command window
clear;
clc;

%% Load the data from .data file and sort them based on indices
data=load('haberman.data');


%% Split and normalize the sorted data into three subsets by using the split scale function
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

% Clustering Per Class
radius=0.5;
% Find the centers (c) of clusters and standard deviations
[c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
[c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
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

% Training 
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);
    
%% Evaluate the TSK models
Y = round(evalfis(tstData(:,1:end-1),valFis));

% Plot the learning curves - Training error and Validation error 
    figure;
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); 
    ylabel('Error');
    legend('Training Error','Validation Error');
    title('TSK model 1: Learning Curves');


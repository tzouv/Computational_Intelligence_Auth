% Script to calcuate the frequency appearance of each class on each subset

%% Calculate the frequency of each class on each subset
% The class ID on the trnData
trn_id = trnData(:, 4);

% Count the occurrences of ID=1 and ID=2
trn_id1 = sum(trn_id == 1);
trn_id2 = sum(trn_id == 2);

% Calculate the percentages
total_data_points = numel(trn_id);
percent_trn1 = (trn_id1 / total_data_points) * 100;
percent_trn2 = (trn_id2 / total_data_points) * 100;

fprintf('Percentage of training data in class 1: %.2f%%\n', percent_trn1);
fprintf('Percentage of training data in class 2: %.2f%%\n', percent_trn2);

% The class ID on the chkData
chk_id = chkData(:, 4);

% Count the occurrences of ID=1 and ID=2
chk_id1 = sum(chk_id == 1);
chk_id2 = sum(chk_id == 2);

% Calculate the percentages
chk_data_points = numel(chk_id);
percent_chk1 = (chk_id1 / chk_data_points) * 100;
percent_chk2 = (chk_id2 / chk_data_points) * 100;

fprintf('Percentage of check data in class 1: %.2f%%\n', percent_chk1);
fprintf('Percentage of check data in class 2: %.2f%%\n', percent_chk2);

% The class ID on the tstData
tst_id = tstData(:, 4);

% Count the occurrences of ID=1 and ID=2
tst_id1 = sum(tst_id == 1);
tst_id2 = sum(tst_id == 2);

% Calculate the percentages
tst_data_points = numel(tst_id);
percent_tst1 = (tst_id1 / tst_data_points) * 100;
percent_tst2 = (tst_id2 / tst_data_points) * 100;

fprintf('Percentage of test data in class 1: %.2f%%\n', percent_tst1);
fprintf('Percentage of test data in class 2: %.2f%%\n', percent_tst2);

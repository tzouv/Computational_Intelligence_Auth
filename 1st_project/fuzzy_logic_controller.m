% Script for design a custom Fuzzy Logic Controller to control the angle of
% a satelite
% Author: Tzouvaras Evangelos
% email: tzouevan@ece.auth.gr


% Create the Fuzzy Inference System
% AND -> min
% ALSO (OR) -> max
% ImplicationMethod -> min, AggregationMethod -> max , Totally -> max-min method(mamdani method)
% Defuzzifier -> COA (Center Of Area)
fis = mamfis('Name', 'MyMamdaniFIS', 'AndMethod', 'min', 'OrMethod', 'max', 'ImplicationMethod', 'min', 'AggregationMethod', 'max', 'DefuzzificationMethod', 'centroid');

% Declare the input variables
fis = addvar(fis, 'input', 'E', [-1,1]);        % The error (E) as input variable
fis = addvar(fis, 'input', 'DE', [-1,1]);       % The derivative error (DE) as input variable

% Declare the output variable
fis = addvar(fis, 'output', 'DU', [-1,1]);      % The derivative control signal (DU) as output variable 

% Declare the fuzzy sets for each linguistic variable

% Fuzzy sets for the 'E' input variable 
fis = addmf(fis, 'input', 1, 'NL', 'trimf', [-1 -1 -0.667]);         % NL set
fis = addmf(fis, 'input', 1, 'NM', 'trimf', [-1 -0.667 -0.333]);     % NM set
fis = addmf(fis, 'input', 1, 'NS', 'trimf', [-0.667 -0.333 0]);      % NS set
fis = addmf(fis, 'input', 1, 'ZR', 'trimf', [-0.333 0 0.333]);       % ZR set
fis = addmf(fis, 'input', 1, 'PS', 'trimf', [0 0.333 0.667]);        % PS set
fis = addmf(fis, 'input', 1, 'PM', 'trimf', [0.333 0.667 1]);        % PM set
fis = addmf(fis, 'input', 1, 'PL', 'trimf', [0.667 1 1]);            % PL set

% Fuzzy sets for the 'DE' input variable 
fis = addmf(fis, 'input', 2, 'NV', 'trimf', [-1 -1 -0.75]);          % NV set
fis = addmf(fis, 'input', 2, 'NL', 'trimf', [-1 -0.75 -0.5]);        % NL set
fis = addmf(fis, 'input', 2, 'NM', 'trimf', [-0.75 -0.5 -0.25]);     % NM set
fis = addmf(fis, 'input', 2, 'NS', 'trimf', [-0.5 -0.25 0]);         % NS set
fis = addmf(fis, 'input', 2, 'ZR', 'trimf', [-0.25 0 0.25]);         % ZR set
fis = addmf(fis, 'input', 2, 'PS', 'trimf', [0 0.25 0.5]);           % PS set
fis = addmf(fis, 'input', 2, 'PM', 'trimf', [0.25 0.5 0.75]);        % PM set
fis = addmf(fis, 'input', 2, 'PL', 'trimf', [0.5 0.75 1]);           % PL set
fis = addmf(fis, 'input', 2, 'PV', 'trimf', [0.75 1 1]);             % PV set

% Fuzzy sets for the 'DU' output variable 
fis = addmf(fis, 'output', 1, 'NL', 'trimf', [-1 -1 -0.667]);        % NL set
fis = addmf(fis, 'output', 1, 'NM', 'trimf', [-1 -0.667 -0.333]);    % NM set
fis = addmf(fis, 'output', 1, 'NS', 'trimf', [-0.667 -0.333 0]);     % NS set
fis = addmf(fis, 'output', 1, 'ZR', 'trimf', [-0.333 0 0.333]);      % ZR set
fis = addmf(fis, 'output', 1, 'PS', 'trimf', [0 0.333 0.667]);       % PS set
fis = addmf(fis, 'output', 1, 'PM', 'trimf', [0.333 0.667 1]);       % PM set
fis = addmf(fis, 'output', 1, 'PL', 'trimf', [0.667 1 1]);           % PL set

% Declare the fuzzy rules: 63 compositions totally
rules = [...
"E==NL & DE==NV => DU=NL"; ...          % 1st row of table
"E==NL & DE==NL => DU=NL"; ...
"E==NL & DE==NM => DU=NL"; ...
"E==NL & DE==NS => DU=NL"; ...
"E==NL & DE==ZR => DU=NL"; ...
"E==NL & DE==PS => DU=NM"; ...
"E==NL & DE==PM => DU=NS"; ...
"E==NL & DE==PL => DU=ZR"; ...
"E==NL & DE==PV => DU=PS"; ...
"E==NM & DE==NV => DU=NL"; ...          % 2nd row of table
"E==NM & DE==NL => DU=NL"; ...
"E==NM & DE==NM => DU=NL"; ...
"E==NM & DE==NS => DU=NL"; ...
"E==NM & DE==ZR => DU=NM"; ...
"E==NM & DE==PS => DU=NS"; ...
"E==NM & DE==PM => DU=ZR"; ...
"E==NM & DE==PL => DU=PS"; ...
"E==NM & DE==PV => DU=PM"; ...
"E==NS & DE==NV => DU=NL"; ...          % 3rd row of table
"E==NS & DE==NL => DU=NL"; ...
"E==NS & DE==NM => DU=NL"; ...
"E==NS & DE==NS => DU=NM"; ...
"E==NS & DE==ZR => DU=NS"; ...
"E==NS & DE==PS => DU=ZR"; ...
"E==NS & DE==PM => DU=PS"; ...
"E==NS & DE==PL => DU=PM"; ...
"E==NS & DE==PV => DU=PL"; ...
"E==ZR & DE==NV => DU=NL"; ...          % 4th row of table
"E==ZR & DE==NL => DU=NL"; ...
"E==ZR & DE==NM => DU=NM"; ...
"E==ZR & DE==NS => DU=NS"; ...
"E==ZR & DE==ZR => DU=ZR"; ...
"E==ZR & DE==PS => DU=PS"; ...
"E==ZR & DE==PM => DU=PM"; ...
"E==ZR & DE==PL => DU=PL"; ...
"E==ZR & DE==PV => DU=PL"; ...
"E==PS & DE==NV => DU=NL"; ...          % 5th row of table
"E==PS & DE==NL => DU=NM"; ...
"E==PS & DE==NM => DU=NS"; ...
"E==PS & DE==NS => DU=ZR"; ...
"E==PS & DE==ZR => DU=PS"; ...
"E==PS & DE==PS => DU=PM"; ...
"E==PS & DE==PM => DU=PL"; ...
"E==PS & DE==PL => DU=PL"; ...
"E==PS & DE==PV => DU=PL"; ...
"E==PM & DE==NV => DU=NM"; ...          % 6th row of table
"E==PM & DE==NL => DU=NS"; ...
"E==PM & DE==NM => DU=ZR"; ...
"E==PM & DE==NS => DU=PS"; ...
"E==PM & DE==ZR => DU=PM"; ...
"E==PM & DE==PS => DU=PL"; ...
"E==PM & DE==PM => DU=PL"; ...
"E==PM & DE==PL => DU=PL"; ...
"E==PM & DE==PV => DU=PL"; ...
"E==PL & DE==NV => DU=NS"; ...          % 7th row of table
"E==PL & DE==NL => DU=ZR"; ...
"E==PL & DE==NM => DU=PS"; ...
"E==PL & DE==NS => DU=PM"; ...
"E==PL & DE==ZR => DU=PL"; ...
"E==PL & DE==PS => DU=PL"; ...
"E==PL & DE==PM => DU=PL"; ...
"E==PL & DE==PL => DU=PL"; ...
"E==PL & DE==PV => DU=PL"; ...
];

% Plot the fuzzy sets for the input varibles
figure(1);
plotmf(fis, 'input', 1, 1000);          % (fis, variable_type, variable_index, datapoints)
figure(2);
plotmf(fis, 'input', 2, 1000);     

% Plot the fuzzy sets for the output variables
figure(3);
plotmf(fis, 'output', 1, 1000);

% Import the rule table into the fis
fis = addRule(fis, rules);

% Write the fis object into a .fis file
writefis(fis, 'fis_satellite.fis');

% This instruction is for graphical presentation of rules 
%ruleview(fis);

% This instuction creates the 3D output graph
gensurf(fis);

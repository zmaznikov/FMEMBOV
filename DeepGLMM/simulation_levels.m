clear
clc

% Set seed for reproducibility
rng(12345); 

% Parameters
I = 100; % Number of individuals
T = 20;  % Number of time periods
num_features = 5; % Number of features

% Generate panel data
features = cell(I, 1); % Cell array to store data for each individual
for i = 1:I
    % Generate T x num_features matrix for individual i
    features{i} = -1 + 2 * rand(T, num_features);
end

% Construct a table to hold the panel data
all_data = []; % To store the flattened data
for i = 1:I
    % Extract features for individual i
    X = features{i};
    
    % Create a time variable
    time = (1:T)';
    
    % Create a group variable
    group = i * ones(T, 1);
    group2 = mod(i, 5) * ones(T, 1) + 1;
    
    % Combine into a table
    individual_data = array2table(X, 'VariableNames', ...
        arrayfun(@(x) sprintf('Feature_%d', x), 1:num_features, 'UniformOutput', false));
    individual_data.Time = time;
    individual_data.Group = group;
    individual_data.Group_2 = group2;
    
    % Append to the complete dataset
    all_data = [all_data; individual_data];
end

% Display the first few rows of the panel data
disp(head(all_data));
%% Transform all_data into a 1x100 cell X

% Initialize the cell array
I = 100; % Number of individuals
X0 = cell(1, I);

% Populate the cell array
for i = 1:I
    % Extract data for individual i
    individual_data = all_data(all_data.Group == i, :);
    
    % Extract features and add the intercept
    features = [ones(size(individual_data, 1), 1), individual_data{:, 1:5}];
    
    % Store in the cell array
    X0{i} = features;
end
%% Split X into train, test and val

% Initialize training, validation, and test cell arrays
X_train0 = cell(1, I);
X_val0 = cell(1, I);
X_test0 = cell(1, I);

% Split the data
for i = 1:I
    % Extract the full matrix for individual i
    data = X0{i};
    
    % Training data: t = 1 to 14
    X_train0{i} = data(1:14, :);
    
    % Validation data: t = 15 to 17
    X_val0{i} = data(15:17, :);
    
    % Test data: t = 18 to 20
    X_test0{i} = data(18:20, :);
end
%% Bernoulli
rng(12345); 
% Preallocate storage
b = normrnd(1, 2.2361, [I, 1]); % Random effects b_i ~ N(1, std)
c = normrnd(1, 2.2361, [5, 1]); % Random effects c_i ~ N(1, std)
epsilon = cell(1, I); % Noise epsilon_{it} for each individual
a = cell(1, I); % a_{it} values
p = cell(1, I); % p_{it} values
y_bern = cell(1, I); % y_{it} values from Bernoulli distribution

% Generate y_{it} for each individual
for i = 1:I
    % Extract X for individual i
    X_i = X0{i}; % X{i} is a 20x6 matrix
    
    % Generate epsilon_{it} ~ N(0, 1) for individual i
    epsilon{i} = normrnd(0, 1, [T, 1]);
    
    % Compute a_{it} using the given formula
    x1 = X_i(:, 2); % Feature 1
    x2 = X_i(:, 3); % Feature 2
    x3 = X_i(:, 4); % Feature 3
    x4 = X_i(:, 5); % Feature 4
    x5 = X_i(:, 6); % Feature 5
    
    % Formula for a_{it}
    a{i} = -1 + 2.5 * x1 ...
           - 2 * x2 ...
           + 2.5 * x3 ...
           - 1.2 * x4 ...
           - 1.3 * x5 ...
           + b(i) + c(mod(i, 5)+1) + epsilon{i};

    % % Formula for a_{it}
    % a{i} = -1 + 2 * log(1 + x1) ...
    %        + (1.1 * x2.^2) ./ exp(x3 + 1) ...
    %        - (1.2 * x4.^3) ./ x5 ...
    %        + b(i) + c(mod(i, 5)+1) + epsilon{i};

    % % Formula for a_{it}
    % a{i} = -3 + 2.5 * log(1 + x1) ...
    %        + (4 * x2.^2) ./ exp(x3) ...
    %        - (2 * x4.^3) ./ x5 ...
    %        + b(i) + c(mod(i, 5)+1) + epsilon{i};
    
    % Compute p_{it} using the logistic function
    p{i} = 1 ./ (1 + exp(-a{i}));
    
    % Generate y_{it} from Bernoulli distribution
    rng(1)
    y_bern{i} = binornd(1, p{i});
end

%% Split y_bern

% Split y_bern into train, val, and test
y_train_bern = cell(1, I);
y_val_bern = cell(1, I);
y_test_bern = cell(1, I);

for i = 1:I
    % Training data: t = 1 to 14
    y_train_bern{i} = y_bern{i}(1:14);
    
    % Validation data: t = 15 to 17
    y_val_bern{i} = y_bern{i}(15:17);
    
    % Test data: t = 18 to 20
    y_test_bern{i} = y_bern{i}(18:20);
end
%% iterate through learning rates

values = [0.1, 0.5, 1]; % The specific values to iterate through
means = values;
vars = values;
for i = 1:length(values)
    current_value = values(i);
    nn = [5,3];
    mdl0_bern = deepGLMMfit(X_train0,y_train_bern,...  
                      X_val0,y_val_bern,...
                      'Distribution','binomial',...
                      'Network',nn,... 
                      'Lrate',current_value,...           
                      'Verbose',1,...             % Display training result each iteration
                      'MaxIter',200,...
                      'Patience',10,...          % Higher patience values could lead to overfitting
                      'S',10,...
                      'Seed',1002);
    MCR_DGLMM = [];
    for s = 1:2000
        rng(s)
        % Make prediction on a test set with true response
        Pred2_bern = deepGLMMpredict(mdl0_bern,X_test0,y_test_bern);                           
        % disp(['PPS on test data                : ', num2str(Pred2_bern.pps)])
        %disp(['Classification rate on test data: ', num2str(Pred2_bern.classification_rate)])
        MCR_DGLMM = [MCR_DGLMM; Pred2_bern.classification_rate];
        means(i) = mean(MCR_DGLMM);
        vars(i) = var(MCR_DGLMM)*10000;
    end
    fprintf('Iteration %d: Value = %.1f\n', i, current_value);
end

%% Alternative grouping

% Initialize the cell array
I = 100; % Number of individuals
X0 = cell(1, 5);

% Populate the cell array
for i = 1:I
    % Extract data for individual i
    individual_data = all_data(all_data.Group_2 == mod(i, 5)+1, :);
    
    % Extract features and add the intercept
    features = [ones(size(individual_data, 1), 1), individual_data{:, 1:5}];
    
    % Store in the cell array
    X0{i} = features;
end

% Initialize training, validation, and test cell arrays
X_train0 = cell(1, 5);
X_val0 = cell(1, 5);
X_test0 = cell(1, 5);

for i = 1:5
    % Extract the full matrix for individual i
    data = X0{i};
    [numRows, numCols] = size(data); % Assuming rows correspond to time indices

    % Training data: t = 0:13, 20:33, etc.
    train_indices = mod(0:numRows-1, 20) <= 13; % Remainder 0-13
    X_train0{i} = data(train_indices, :);

    % Validation data: t = 14:16, 34:36, etc.
    val_indices = mod(0:numRows-1, 20) >= 14 & mod(0:numRows-1, 20) <= 16; % Remainder 14-16
    X_val0{i} = data(val_indices, :);

    % Test data: t = 17:19, 37:39, etc.
    test_indices = mod(0:numRows-1, 20) >= 17; % Remainder 17-19
    X_test0{i} = data(test_indices, :);
end





%% Format p for GLMM

% Preallocate a vector to hold all responses
true_p = []; % This will store all y values from the cell array

% Combine all responses from the y cell array
for i = 1:numel(p) % Loop over each individual
    true_p = [true_p; p{i}]; % Append the 20x1 response vector to the list
end

% Add the response column to the all_data table
all_data.True_p = true_p;

% Filter rows where Time <= 14
glmm_training_data = all_data(all_data.Time <= 17, :);

%%
figure
histogram(all_data.True_p)

%% Format y for GLMM

% Preallocate a vector to hold all responses
response = []; % This will store all y values from the cell array

% Combine all responses from the y cell array
for i = 1:numel(y_bern) % Loop over each individual
    response = [response; y_bern{i}]; % Append the 20x1 response vector to the list
end

% Add the response column to the all_data table
all_data.Response = response;

% Filter rows where Time <= 14
glmm_training_data = all_data(all_data.Time <= 17, :);

%% Alternative y formatting

% Initialize cell arrays
y_train_bern = cell(1, 5);
y_val_bern = cell(1, 5);
y_test_bern = cell(1, 5);

% Loop through each Group_2
for group = 1:5
    % Select rows belonging to the current Group_2
    group_data = all_data(all_data.Group_2 == group, :);
    
    % Train: Time <= 14
    train_data = group_data(group_data.Time <= 14, :);
    y_train_bern{group} = train_data.Response; % Extract Feature_5 as doubles
    
    % Validation: Time 15–17
    val_data = group_data(group_data.Time >= 15 & group_data.Time <= 17, :);
    y_val_bern{group} = val_data.Response; % Extract Feature_5 as doubles
    
    % Test: Time 18–20
    test_data = group_data(group_data.Time >= 18 & group_data.Time <= 20, :);
    y_test_bern{group} = test_data.Response; % Extract Feature_5 as doubles
end


%% Fit deepGLMM model using default setting on y_bern
% By default, if 'distribution' option is not specified then deepGLMMfit
% will assign the response variables as 'normal'
nn = [5,5];
mdl0_bern = deepGLMMfit(X_train0,y_train_bern,...  
                  X_val0,y_val_bern,...
                  'Distribution','binomial',...
                  'Network',nn,... 
                  'Lrate',1,...           
                  'Verbose',1,...             % Display training result each iteration
                  'MaxIter',100,...
                  'Patience',10,...          % Higher patience values could lead to overfitting
                  'S',10,...
                  'Seed',1001);
%%
glme = fitglme(glmm_training_data,...
'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + (1|Group) + (1|Group_2)',...
'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
'DummyVarCoding','effects');

glme

%% 

mufit = fitted(glme);

figure
scatter(glmm_training_data.True_p,mufit)
title('Observed Values versus Fitted Values')
xlabel('True Values')
ylabel('Fitted Values')
%%
glmm_test_data = all_data(all_data.Time > 17, :);

%%
table(predict(glme, glmm_test_data), random('Binomial', 1, predict(glme, glmm_test_data)))

%%
MCR = [];
for i = 1:2000
    rng(i)
    Results = table(predict(glme, glmm_test_data), random('Binomial', 1, predict(glme, glmm_test_data)), glmm_test_data.Response);
    % Results
    MCR = [MCR; mean(abs(Results.Var2-Results.Var3))];
end

%% CART

% Specify predictor and response column names
predictorColumns = {'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'}; 
responseColumn = 'Response';

% Filter the predictors and response from the table
X = glmm_training_data(:, predictorColumns); % Select predictors
Y = glmm_training_data.(responseColumn);    % Select response as a vector

tree = fitctree(X, Y);

%%
% View the tree
view(tree, 'Mode', 'graph');

%%

%predictedClass = predict(tree, glmm_test_data(:, predictorColumns));
Results = table(predict(tree, glmm_test_data(:, predictorColumns)), glmm_test_data.Response);
mean(abs(Results.Var1-Results.Var2))

%%
writetable(all_data, '../Data/nonlin-3_levels_s12345.csv');
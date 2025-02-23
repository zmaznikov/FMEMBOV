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
    
    % Combine into a table
    individual_data = array2table(X, 'VariableNames', ...
        arrayfun(@(x) sprintf('Feature_%d', x), 1:num_features, 'UniformOutput', false));
    individual_data.Time = time;
    individual_data.Group = group;
    
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

% Preallocate storage
b = normrnd(1, 0.3162, [I, 1]); % Random effects b_i ~ N(1, std)
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
           + b(i) + epsilon{i};
    
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

%% Prediction on y_bern test data
% Make prediction (point estimation) on a test set
% Make prediction on a test set without true response
rng(1)
Pred1_bern = deepGLMMpredict(mdl0_bern,X_test0); 

rng(1)
% Make prediction on a test set with true response
Pred2_bern = deepGLMMpredict(mdl0_bern,X_test0,y_test_bern);                           
disp(['PPS on test data                : ', num2str(Pred2_bern.pps)])
disp(['Classification rate on test data: ', num2str(Pred2_bern.classification_rate)])

%% different seeds
MCR_DGLMM = [];
for i = 1:2000
    rng(i)
    % Make prediction on a test set with true response
    Pred2_bern = deepGLMMpredict(mdl0_bern,X_test0,y_test_bern);                           
    % disp(['PPS on test data                : ', num2str(Pred2_bern.pps)])
    %disp(['Classification rate on test data: ', num2str(Pred2_bern.classification_rate)])
    MCR_DGLMM = [MCR_DGLMM; Pred2_bern.classification_rate];
end

%%
figure
histogram(MCR_DGLMM)

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
                      'MaxIter',100,...
                      'Patience',10,...          % Higher patience values could lead to overfitting
                      'S',10,...
                      'Seed',1001);
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


%% GLMM

FEglme = fitglme(glmm_training_data,...
'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5',...
'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
'DummyVarCoding','effects');

FEglme

glme = fitglme(glmm_training_data,...
'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + (1|Group)',...
'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
'DummyVarCoding','effects');

glme

[psi,dispersion,stats] = covarianceParameters(glme);
stats{1}

results = compare(FEglme,glme,'CheckNesting',true)
%% 

mufit = fitted(glme);

% figure
% scatter(glmm_training_data.Response,mufit)
% title('Observed Values versus Fitted Values')
% xlabel('Fitted Values')
% ylabel('Observed Values')

figure
scatter(glmm_training_data.True_p,mufit)
title('Observed Values versus Fitted Values')
xlabel('True Values')
ylabel('Fitted Values')
%% 

plotResiduals(glme,'histogram','ResidualType','Pearson')
%% 
plotResiduals(glme,'fitted','ResidualType','Pearson')
%% 
plotResiduals(glme,'lagged','ResidualType','Pearson')

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

%%
figure
histogram(MCR)

%%
MCR_FE = [];
for i = 1:200
    rng(i)
    Results = table(predict(FEglme, glmm_test_data), random('Binomial', 1, predict(FEglme, glmm_test_data)), glmm_test_data.Response);
    % Results
    MCR_FE = [MCR_FE; mean(abs(Results.Var2-Results.Var3))];
end

histogram(MCR_FE)

%% DeepGLM data prep
selected_columns = {'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'};
X_train_deepglm = table2array(all_data(all_data.Time <= 17, selected_columns));
y_train_deepglm = table2array(all_data(all_data.Time <= 17, {'Response'}));

X_test_deepglm = table2array(all_data(all_data.Time > 17, selected_columns));
y_test_deepglm = table2array(all_data(all_data.Time > 17, {'Response'}));


%% Fit deepGLM model using default setting
nn2 = [5,5];
mdl_DGLM = deepGLMfit(X_train_deepglm,y_train_deepglm,... 
                 'Distribution','binomial',...
                 'Network',nn2,... 
                 'Lrate',0.01,...           
                 'Verbose',1,...             % Display training result each iteration
                 'BatchSize',size(X_train_deepglm,1),...   % Use entire training data as mini-batch
                 'MaxEpoch',10000,...
                 'Patience',50,...           % Higher patience values could lead to overfitting
                 'Seed',100);
%% Plot training output    
% Plot lowerbound
figure
plot(mdl_DGLM.out.lbBar,'LineWidth',2)
title('Lowerbound of Variational Approximation','FontSize',20)
xlabel('Iterations','FontSize',14,'FontWeight','bold')
ylabel('Lowerbound','FontSize',14,'FontWeight','bold')
grid on

% Plot shrinkage coefficients
figure
deepGLMplot('Shrinkage',mdl_DGLM.out.shrinkage,...
            'Title','Shrinkage Coefficients',...
            'Xlabel','Iterations',...
            'LineWidth',2);

%% Prediction on test data
% Make prediction (point estimation) on a test set
Pred1 = deepGLMpredict(mdl_DGLM,X_test_deepglm);

% If ytest is specified (for model evaluation purpose)
% then we can check PPS and MSE on test set
Pred2 = deepGLMpredict(mdl_DGLM,X_test_deepglm,'ytest',y_test_deepglm);
disp(['PPS on test data: ',num2str(Pred2.pps)])
disp(['Classification rate on test data: ',num2str(Pred2.accuracy)])

% Plot ROC curve
figure
deepGLMplot('ROC',Pred2.yProb,...
            'ytest',y_test_deepglm,...
            'Title','ROC',...
            'Xlabel','False Positive Rate',...
            'Ylabel','True Positive Rate')

%% different seeds
MCR_DGLM = [];
for i = 1:2000
    rng(i)
    % Make prediction on a test set with true response
    Pred2 = deepGLMpredict(mdl_DGLM,X_test_deepglm,'ytest',y_test_deepglm);
    MCR_DGLM = [MCR_DGLM; mean(abs(y_test_deepglm - random('Binomial', 1, Pred2.yProb)))];
end

%% Compare to linear model
figure
mdlLR = fitglm(X_train_deepglm,y_train_deepglm,'Distribution','binomial','Link','logit');
yProb = predict(mdlLR,X_test_deepglm);
deepGLMplot('ROC',[Pred2.yProb,yProb],...
            'ytest',y_test_deepglm,...
            'Title','ROC',...
            'Xlabel','False Positive Rate',...
            'Ylabel','True Positive Rate',...
            'legend',{'deepGLM','Logistic Regression'})

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
writetable(all_data, '../Data/linear-1_var0.1_s12345.csv');
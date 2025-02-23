clear
clc

folder_path = '../Data';
destination_folder = 'TrueP';

% Get a list of all files in the directory
file_list = dir(folder_path);

% Loop through each file in the directory
for i = 1:length(file_list)
    % Skip '.' and '..' entries
    if file_list(i).isdir
        continue; % Skip directories
    end
    
    % Get the full file path
    file_path = fullfile(folder_path, file_list(i).name);

    if ~contains(file_path, 'dglmm') 
    
        % Process the file
        fprintf('Processing file: %s\n', file_path);
        
        all_data = readtable(file_path);

        fig = figure;
        h = histogram(all_data.True_p); 
        h.BinWidth = 0.1;

        % Set y-axis limits
        ylim([0 1000]);

        % Customize the axes appearance
        set(gca, 'FontSize', 14);    % Set font size

        fig.Units = 'centimeters';         
        fig.OuterPosition = [0, 0, 14, 10];   % [x, y, width, height]

        % Get the name of the file without the extension
        [~, name, ~] = fileparts(file_path);

        % Create the new file path with a .pdf extension in the destination folder
        new_file_path = fullfile(destination_folder, [name, '.pdf']);

        % Export the figure to a PDF file
        exportgraphics(fig, new_file_path); 
    end
end


%% just plot for non-GLMM predictions
folder_path = '../Data/TrueAndPredictedTest';
destination_folder = 'TrueAndPredictedTest';

% Get a list of all files in the directory
file_list = dir(folder_path);

% Loop through each file in the directory
for i = 1:length(file_list)
    % Skip '.' and '..' entries
    if file_list(i).isdir
        continue; % Skip directories
    end
    
    % Get the full file path
    file_path = fullfile(folder_path, file_list(i).name);


    % Process the file
    fprintf('Processing file: %s\n', file_path);
    
    opts = detectImportOptions(file_path);  % Automatically detect options
    opts.VariableNamingRule = 'preserve';   % Set to preserve original column names
    
    p_data = readtable(file_path);

    fig = figure('Visible', 'off');
    if contains(file_path, 'LMMNN') 
        scatter(p_data.True_p, p_data.Predicted_p, 25, 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
    else
        scatter(p_data.(1), p_data.(2), 25, 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
    end
    % title('Observed Values versus Fitted Values')
    xlabel('True p')
    ylabel('Predicted p')
    box on;
    
    % Customize the axes appearance
    set(gca, 'FontSize', 14);    % Set font size

    fig.Units = 'centimeters';         
    fig.OuterPosition = [0, 0, 14, 10];   % [x, y, width, height]

    % Get the name of the file without the extension
    [~, name, ~] = fileparts(file_path);
    
    % Create the new file path with a .pdf extension in the destination folder
    new_file_path = fullfile(destination_folder, [name, '.pdf']);
    
    % Export the figure to a PDF file
    exportgraphics(fig, new_file_path); 

end
%% GLMM predict and plot
folder_path = '../Data';
destination_folder = 'TrueAndPredictedTest';

% Get a list of all files in the directory
file_list = dir(folder_path);

% Loop through each file in the directory
for i = 1:length(file_list)
    % Skip '.' and '..' entries
    if file_list(i).isdir
        continue; % Skip directories
    end
    
    % Get the full file path
    file_path = fullfile(folder_path, file_list(i).name);

    % Process the file
    fprintf('Processing file: %s\n', file_path);
    
    all_data = readtable(file_path);

    glmm_training_data = all_data(all_data.Time <= 17, :);
    glmm_test_data = all_data(all_data.Time > 17, :);

    if contains(file_path, 'lmmnn') 
        glme = fitglme(glmm_training_data,...
        'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + Feature_6 + Feature_7 + Feature_8 + Feature_9 + Feature_10 + (1|Group)',...
        'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
        'DummyVarCoding','effects');
    elseif ~contains(file_path, 'dglmm')
        glme = fitglme(glmm_training_data,...
        'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + (1|Group)',...
        'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
        'DummyVarCoding','effects');
    end

    if ~contains(file_path, 'dglmm')
        % Set seed for reproducibility
        rng(12345);
        p_data = table(glmm_test_data.True_p, predict(glme, glmm_test_data));

        fig = figure('Visible', 'off');
        scatter(p_data.(1),p_data.(2), 25, 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
        % title('Observed Values versus Fitted Values')
        xlabel('True p')
        ylabel('Predicted p')
        box on;
        
        % Customize the axes appearance
        set(gca, 'FontSize', 14);    % Set font size
        
        fig.Units = 'centimeters';         
        fig.OuterPosition = [0, 0, 14, 10];   % [x, y, width, height]
        
        % Get the name of the file without the extension
        [~, name, ~] = fileparts(file_path);
        
        % Create the new file path with a .pdf extension in the destination folder
        new_file_path = fullfile(destination_folder, ['GLMM', name, '.pdf']);
        
        writetable(p_data, fullfile('R/TrueAndPredictedTest', ['GLMM', name, '.csv']));
        % Export the figure to a PDF file
        exportgraphics(fig, new_file_path); 

        if contains(file_path, 'levels') 
            glme = fitglme(glmm_training_data,...
            'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + (1|Group_2)',...
            'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
            'DummyVarCoding','effects');

            % Set seed for reproducibility
            rng(12345);
            p_data = table(glmm_test_data.True_p, predict(glme, glmm_test_data));
    
            fig = figure('Visible', 'off');
            scatter(p_data.(1),p_data.(2), 25, 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
            % title('Observed Values versus Fitted Values')
            xlabel('True p')
            ylabel('Predicted p')
            box on;
            
            % Customize the axes appearance
            set(gca, 'FontSize', 14);    % Set font size
            
            fig.Units = 'centimeters';         
            fig.OuterPosition = [0, 0, 14, 10];   % [x, y, width, height]
            
            % Get the name of the file without the extension
            [~, name, ~] = fileparts(file_path);
            
            % Create the new file path with a .pdf extension in the destination folder
            new_file_path = fullfile(destination_folder, ['GLMM', name, '_group', '.pdf']);
            
            writetable(p_data, fullfile('R/TrueAndPredictedTest', ['GLMM', name, '_group', '.csv']));
            % Export the figure to a PDF file
            exportgraphics(fig, new_file_path); 

            glme = fitglme(glmm_training_data,...
            'Response ~ Feature_1 + Feature_2 + Feature_3 + Feature_4 + Feature_5 + (1|Group) + (1|Group_2)',...
            'Distribution','Binomial','Link','logit','FitMethod','Laplace',...
            'DummyVarCoding','effects');

            % Set seed for reproducibility
            rng(12345);
            p_data = table(glmm_test_data.True_p, predict(glme, glmm_test_data));
    
            fig = figure('Visible', 'off');
            scatter(p_data.(1),p_data.(2), 25, 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0.25);
            % title('Observed Values versus Fitted Values')
            xlabel('True p')
            ylabel('Predicted p')
            box on;
            
            % Customize the axes appearance
            set(gca, 'FontSize', 14);    % Set font size
            
            fig.Units = 'centimeters';         
            fig.OuterPosition = [0, 0, 14, 10];   % [x, y, width, height]
            
            % Get the name of the file without the extension
            [~, name, ~] = fileparts(file_path);
            
            % Create the new file path with a .pdf extension in the destination folder
            new_file_path = fullfile(destination_folder, ['GLMM', name, '_both', '.pdf']);
            
            writetable(p_data, fullfile('R/TrueAndPredictedTest', ['GLMM', name, '_both', '.csv']));
            % Export the figure to a PDF file
            exportgraphics(fig, new_file_path); 
        end
    end


    
end
function main(varargin)

addpath("layers")
addpath("utility")

clc;

if nargin < 1
    error("Pass at least 1 config file as argument");
end

for i=1:nargin
    config_name = varargin{i};
    config_file = read_config(config_name);

    n_configs = size(config_file, 1);
    for j=1:n_configs
        config = config_file(j);
        % config.timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-ss');
        config.runID = sprintf("%s-%s-%s", config.dataset, config.encoder(end).layerType, datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-ss'));
    
        % Load Dataset
        [XTrain, YTrain, XTest, YTest] = loadDataset(config.dataset);

        % Build Model
        [netE, netD] = buildModel(config);

        % Train Model
        [netE, netD, trainStats] = train(netE, netD, XTrain, config);

        dumpModel(config, trainStats, netE, netD);
        stats = generateStatistics(config, trainStats);

        % Test Model
        avgTestLoss = test(netE, netD, XTest, YTest, config);
        stats.testLoss = avgTestLoss;
    
        saveStatistics(stats);
    end
end
end



%%% ======= UTILITY FUNCTIONS ======= %%%

% read config file
function config = read_config(filename)
    % Check if the file exists
    if exist(filename, 'file') ~= 2
        error('File not found: %s', filename);
    end

    % Read the JSON file
    fid = fopen(filename, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);

    % Parse the JSON string into a MATLAB structure
    config = jsondecode(str);
end

% generate run statistics from config
function stats = generateStatistics(config, trainStats)
    
    stats = struct;
    stats.runID = config.runID;
    stats.dataset = config.dataset;

    stats.samplingLayer = config.encoder(end).layerType;
    stats.numLatentChannels = config.numLatentChannels;
    stats.numEncoderLayers = countLayers(config.encoder);
    stats.numDecoderLayers = countLayers(config.decoder);

    stats.numEpochs =  config.numEpochs;
    stats.learningRate = config.learningRate;
    stats.batchSize = config.batchSize;

    if isfield(config, "notes")
        stats.notes = config.notes;
    end
    stats.avgEpochTime = trainStats.avgEpochTime;
    stats.finalLoss = trainStats.finalLoss;  
    
    function n = countLayers(net)
        n = 0;
        for i = 1:length(net)
            layer = net(i);
            if strcmp(layer.layerType, "fullyConnectedLayer")
                n = n+1;
            end
        end
    end
end


% save config statistics to disk
function [] = saveStatistics(stats)
    stats_filename = "statistics.csv";
    if exist(stats_filename, 'file') == 0
        % If it doesn't exist, create it with headers
        headers = fieldnames(stats);
        fid = fopen(stats_filename, 'w');
        fprintf(fid, '%s,', headers{1:end-1});
        fprintf(fid, '%s\n', headers{end});
        fclose(fid);
    end
    
    % Open the CSV file to append the content of the struct
    fid = fopen(stats_filename, 'a');
    values = struct2cell(stats);
    values = cellfun(@num2str, values, 'UniformOutput', false);
    fprintf(fid, '%s,', values{1:end-1});
    fprintf(fid, '%s\n', values{end});
    fclose(fid);
end

% save trained model to disk
function dumpModel(config, trainStats, netE, netD)
    baseFolder = "model_dumps";  
    if exist(baseFolder, 'dir') == 0
        mkdir(baseFolder);
    end
    
    modelFolder = sprintf("%s/%s", baseFolder, config.runID);
    mkdir(modelFolder);

    plot_filename = sprintf("%s/training_loss.png", modelFolder);
    fig = figure;
    plot(trainStats.lossHistory);
    xlabel('Iterations');
    ylabel('Loss');
    title('Training Loss');
    % Save plot as an image file
    saveas(fig, plot_filename);

    model_filename = sprintf("%s/model.mat", modelFolder);
    loss_filename = sprintf("%s/loss.mat", modelFolder);
    
    save(model_filename, "netE", "netD", "config");
    lossHistory = extractdata(trainStats.lossHistory);
    save(loss_filename, "lossHistory");

    config_filename = sprintf("%s/config.mat", modelFolder);
    save(config_filename, "config");
end
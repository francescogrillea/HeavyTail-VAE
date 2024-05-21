addpath("layers")

clc; clear;

profile off
profile on

% Load Dataset
dataset = load('mnist.mat');

XTrain = reshape(dataset.training.images, 28,28,1,[]);
XTest = reshape(dataset.test.images, 28,28,1,[]);

sizeXTrain = size(XTrain);
imageSize = sizeXTrain(1:end-1);

% iterate through different configs
configs = read_config("config.json");
n_configs = size(configs, 1);
for i=1:n_configs
    config = configs(i);
    config.timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-ss');

    % Build Model
    [netE, netD] = buildModel(imageSize, config);

    % Train Model
    [netE, netD, trainStats] = train(netE, netD, XTrain, config);
        
    stats = generateStatistics(config, trainStats);
    saveStatistics(stats);
    dumpModel(config, trainStats, netE, netD);

    % Test Model
    % test(netE, netD, config.numLatentChannels, XTest(:,:,:,1:5));
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
    stats.timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-ss');
    stats.samplingLayer = config.samplingLayer;
    stats.numLatentChannels = config.numLatentChannels;
    stats.numEncoderLayers = config.encoder.nHidden;
    stats.numDecoderLayers = config.decoder.nHidden;

    stats.numEpochs =  config.numEpochs;
    stats.learningRate = config.learningRate;
    stats.batchSize = config.batchSize;

    if isfield(config, "notes")
        stats.notes = config.notes;
    end
    stats.avgEpochTime = trainStats.avgEpochTime;
    stats.finalLoss = trainStats.finalLoss;  
    
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
function [] = dumpModel(config, trainStats, netE, netD)
    baseFolder = "model_dumps";  
    if exist(baseFolder, 'dir') == 0
        mkdir(baseFolder);
    end
    
    modelFolder = sprintf("%s/%s", baseFolder, config.timestamp);
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
    
    save(model_filename, "netE", "netD");
end
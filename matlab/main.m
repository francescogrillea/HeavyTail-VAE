clc; clear;

% Config
config = struct;
config.timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HH-MM-ss');
config.numLatentChannels = 10;
config.neuronsPerLayer = 300;
config.hiddenLayersEncoder = 3;
config.hiddenLayersDecoder = 4;
config.sampleDistribution = "LogNormal";
config.numEpochs = 10;
config.learningRate = 1e-5;
config.batchSize = 12;
config.plotLoss = true;


% Load Dataset
dataset = load('mnist.mat');

XTrain = reshape(dataset.training.images, 28,28,1,[]);
XTest = reshape(dataset.test.images, 28,28,1,[]);

sizeXTrain = size(XTrain);
imageSize = sizeXTrain(1:end-1);

% Build Model
[netE, netD] = buildModel(imageSize, config);

% Train Model
[netE, netD] = train(netE, netD, XTrain, config);

saveStatistics(config);
dumpModel(config, netE, netD);

% Test Model
% test(netE, netD, config.numLatentChannels, XTest(:,:,:,1:5));


function [] = saveStatistics(config)
    stats_filename = "statistics.csv";
    if exist(stats_filename, 'file') == 0
        % If it doesn't exist, create it with headers
        headers = fieldnames(config);
        fid = fopen(stats_filename, 'w');
        fprintf(fid, '%s,', headers{1:end-1});
        fprintf(fid, '%s\n', headers{end});
        fclose(fid);
    end
    
    % Open the CSV file to append the content of the struct
    fid = fopen(stats_filename, 'a');
    values = struct2cell(config);
    values = cellfun(@num2str, values, 'UniformOutput', false);
    fprintf(fid, '%s,', values{1:end-1});
    fprintf(fid, '%s\n', values{end});
    fclose(fid);
end

function [] = dumpModel(config, netE, netD)
    % TODO - add loss plot
    baseFolder = "model_dumps";  
    if exist(baseFolder, 'dir') == 0
        mkdir(baseFolder);
    end
    
    modelFolder = sprintf("%s/%s", baseFolder, config.timestamp);
    mkdir(modelFolder);
    filename = sprintf("%s/model.mat", modelFolder);
    
    save(filename, "netE", "netD");
end
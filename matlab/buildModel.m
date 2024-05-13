function [encoder, decoder] = buildModel(imageSize, config)

    if nargin < 2
        config = struct;
    end
       
    if ~isfield(config, "numLatentChannels")
        fprintf("Setting default numLatenteChannels = 10.\n");
        config.numLatentChannels = 10;
    end
    if ~isfield(config, "neuronsPerLayer")
        fprintf("Setting default neuronsPerLayer = 300.\n");
        config.neuronsPerLayer = 300;
    end
    if ~isfield(config, "hiddenLayersEncoder") || config.hiddenLayersEncoder < 2
        fprintf("Setting default hiddenLayersEncoder = 3.\n");
        config.hiddenLayersEncoder = 3;
    else
        config.hiddenLayersEncoder = config.hiddenLayersEncoder - 2;
    end
    if ~isfield(config, "hiddenLayersDecoder") || config.hiddenLayersDecoder < 2
        fprintf("Setting default hiddenLayersEncoder = 4.\n");
        config.hiddenLayersDecoder = 4;
    else
        config.hiddenLayersDecoder = config.hiddenLayersDecoder - 2;
    end
    if ~isfield(config, "sampleDistribution")
        fprintf("Setting default sampleDistribution = Normal.\n");
        config.sampleDistribution = "Normal";
    end

    % Encoder
    layersE = [
        imageInputLayer(imageSize, Normalization="rescale-zero-one")
        fullyConnectedLayer(config.neuronsPerLayer, Name="eInputLayer")
        leakyReluLayer(0.01)
    ];    
    for i=1:config.hiddenLayersEncoder
        layersE = [
            layersE
            fullyConnectedLayer(config.neuronsPerLayer, Name="eHidden"+i)
            leakyReluLayer(0.01)
        ];
    end
    
    samplingLayer = setSamplingLayer(config.sampleDistribution);

    layersE = [layersE
        fullyConnectedLayer(config.numLatentChannels*2, Name="eOutputLayer")
        samplingLayer
    ];


    % Decoder
    layersD = [
        featureInputLayer(config.numLatentChannels)
        fullyConnectedLayer(config.neuronsPerLayer, Name="dInputLayer")
        leakyReluLayer(0.01)
    ];
    for i=1:config.hiddenLayersDecoder
        layersD = [
            layersD
            fullyConnectedLayer(config.neuronsPerLayer, Name="dHidden"+i)
            leakyReluLayer(0.01)
        ];
    end
    
    layersD = [
        layersD
        fullyConnectedLayer(prod(imageSize), Name="dOutputLayer")
        sigmoidLayer
        reshapeLayer(imageSize)
    ];

    encoder = dlnetwork(layersE);
    decoder = dlnetwork(layersD);

end

function samplingLayer = setSamplingLayer(distr)
    
    if strcmp(distr, "Normal") == 1
        samplingLayer = normalSamplingLayer;
    elseif strcmp(distr, "LogNormal") == 1
        samplingLayer = logNormalSamplingLayer; 
    else
        fprintf("No distribution named %s. Using Normal.\n", distr);
        samplingLayer = normalSamplingLayer;
    end

end
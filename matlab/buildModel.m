function [encoder, decoder] = buildModel(imageSize, config)

    if nargin < 2
        config = struct;
    end
       
    if ~isfield(config, "numLatentChannels")
        config.numLatentChannels = 10;
    end
    if ~isfield(config, "neuronsPerLayer")
        config.neuronsPerLayer = 300;
    end
    if ~isfield(config, "hiddenLayersEncoder")
        config.hiddenLayersEncoder = 3;
    else
        config.hiddenLayersEncoder = config.hiddenLayersEncoder - 2;
    end
    if ~isfield(config, "hiddenLayersDecoder")
        config.hiddenLayersDecoder = 4;
    else
        config.hiddenLayersDecoder = config.hiddenLayersDecoder - 2;
    end
    if ~isfield(config, "distribution")
        config.distribution = "Normal";
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
    
    if strcmp(config.distribution, "Normal") == 1
        samplingLayer = normalSamplingLayer;
    elseif strcmp(config.distribution, "LogNormal") == 1
        samplingLayer = logNormalSamplingLayer; 
    else
        fprintf("No distribution named %s. Using Normal.\n", confnig.distribution)
        samplingLayer = normalSamplingLayer;
    end

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
function [encoder, decoder] = buildModel(imageSize, config)

    % default config parameters
    DEFAULT_N_LATENT_CHANNELS = 10;

    if nargin < 2
        config = struct;
    end
       
    if ~isfield(config, "numLatentChannels")
        fprintf("Setting default numLatenteChannels = %.\n", DEFAULT_N_LATENT_CHANNELS);
        config.numLatentChannels = 10;
    end



    % if ~isfield(config, "neuronsPerLayer")
    %     fprintf("Setting default neuronsPerLayer = 300.\n");
    %     config.neuronsPerLayer = 300;
    % end
    % if ~isfield(config, "hiddenLayersEncoder") || config.hiddenLayersEncoder < 2
    %     fprintf("Setting default hiddenLayersEncoder = 3.\n");
    %     config.hiddenLayersEncoder = 3;
    % else
    %     config.hiddenLayersEncoder = config.hiddenLayersEncoder - 2;
    % end
    % if ~isfield(config, "hiddenLayersDecoder") || config.hiddenLayersDecoder < 2
    %     fprintf("Setting default hiddenLayersEncoder = 4.\n");
    %     config.hiddenLayersDecoder = 4;
    % else
    %     config.hiddenLayersDecoder = config.hiddenLayersDecoder - 2;
    % end
    % if ~isfield(config, "sampleDistribution")
    %     fprintf("Setting default sampleDistribution = Normal.\n");
    %     config.sampleDistribution = "Normal";
    % end

    % Encoder
    layersE = [
        imageInputLayer(imageSize, Normalization="rescale-zero-one")
    ];
    for i=1:config.encoder.nHidden
        if i == config.encoder.nHidden 
            neurons = config.numLatentChannels;
        else
            neurons = config.encoder.layers(i).neurons;
        end
        layerType = feval(config.encoder.layers(i).layerType, neurons, Name="eLayer"+i);
        layersE = [
            layersE
            layerType
            leakyReluLayer(0.01)
        ];
    end
    % add sampling layer
    samplingLayer = feval(config.samplingLayer, Name="SamplingLayer");
    layersE = [layersE
        samplingLayer
    ];


    % Decoder
    layersD = [
        featureInputLayer(config.numLatentChannels)
    ];
    for i=1:config.decoder.nHidden
        if i == config.decoder.nHidden
            neurons = prod(imageSize);
            activation = sigmoidLayer;
        else
            neurons = config.decoder.layers(i).neurons;
            activation = leakyReluLayer(0.01);
        end
        layerType = feval(config.decoder.layers(i).layerType, neurons, Name="dLayer"+i);
        layersD = [
            layersD
            layerType
            activation
        ];
    end

    layersD = [
        layersD
        reshapeLayer(imageSize)
    ];

    encoder = dlnetwork(layersE);
    decoder = dlnetwork(layersD);

end
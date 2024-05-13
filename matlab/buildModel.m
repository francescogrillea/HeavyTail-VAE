function [encoder, decoder] = buildModel(config, imageSize)

    numLatentChannels = config.numLatentChannels;
    neuronsPerLayer = config.neuronsPerLayer;
    hiddenLayersEncoder = config.hiddenLayersEncoder - 2;
    hiddenLayersDecoder = config.hiddenLayersDecoder - 2;

    % Encoder
    layersE = [
        imageInputLayer(imageSize, Normalization="rescale-zero-one")
        fullyConnectedLayer(neuronsPerLayer, Name="eInputLayer")
        leakyReluLayer(0.01)
    ];    
    for i=1:hiddenLayersEncoder
        layersE = [
            layersE
            fullyConnectedLayer(neuronsPerLayer, Name="eHidden"+i)
            leakyReluLayer(0.01)
        ];
    end

    layersE = [layersE
        fullyConnectedLayer(numLatentChannels*2, Name="eOutputLayer")
        logNormalSamplingLayer
    ];


    % Decoder
    layersD = [
        featureInputLayer(numLatentChannels)
        fullyConnectedLayer(neuronsPerLayer, Name="dInputLayer")
        leakyReluLayer(0.01)
    ];
    for i=1:hiddenLayersDecoder
        layersD = [
            layersD
            fullyConnectedLayer(neuronsPerLayer, Name="dHidden"+i)
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
function [encoder, decoder] = buildModel(config)

    if nargin < 1
        error("Must provide a config of the network");
    end
    
    config.inputSize = reshape(config.inputSize, 1, []);

    % Encoder
    layersE = [];

    for i = 1:length(config.encoder)
        layerConfig = config.encoder{i};
        layersE = [layersE
            buildLayer(layerConfig);
        ];
    end

    % Decoder
    layersD = [];
    for i = 1:length(config.decoder)
        layerConfig = config.decoder{i};
        layersD = [layersD
            buildLayer(layerConfig);
        ];
    end

    encoder = dlnetwork(layersE);
    decoder = dlnetwork(layersD);

    
    function layer = buildLayer(layerConfig)
        layerType = layerConfig.layerType;
    
        args = {};
        argc = 1;
        if isfield(layerConfig, "defaultArg")
            if ischar(layerConfig.defaultArg)
                args{1} = eval(layerConfig.defaultArg);
            else
                args{1} = layerConfig.defaultArg;
            end
            argc = argc + 1;
        end

        configFields = fields(layerConfig);
        for f = 1:length(configFields)
            field = configFields{f};

            if ~strcmp(field, "layerType") && ~strcmp(field, "defaultArg")
                args{argc} = field;
                if ischar(layerConfig.(field))
                    args{argc+1} = eval(layerConfig.(field));
                else
                    args{argc+1} = layerConfig.(field);
                end
                argc = argc + 2;
            end
        end

        layer = feval(layerType, args{:}, "Name", layerType);
    end
end
function [encoder, decoder] = buildModel(config)

    if nargin < 1
        error("Must provide a config of the network");
    end
    
    config.inputSize = reshape(config.inputSize, 1, []);

    % Encoder
    layersE = [];

    for i = 1:length(config.encoder)
        layerConfig = config.encoder(i);
        layersE = [layersE
            buildLayer(layerConfig);
        ];
    end

    % Decoder
    layersD = [];
    for i = 1:length(config.decoder)
        layerConfig = config.decoder(i);
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
        if isfield(layerConfig, "defaultArg") && ~isempty(layerConfig.defaultArg)
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

            if ~strcmp(field, "layerType") && ~strcmp(field, "defaultArg") && ~isempty(layerConfig.(field))
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

    function structarray = mergedissimilarstructures(structures, defaultempty)
        %structures: a cell array of scalar structures to merge
        %defaultempy: the value to use to fill missing fields. Optional, default = []
        %structarray: a structure array the same size as the input structures cell array. 
        % The fields of structurarray is the union of the fields of the input structures
        
        %TODO: input validation
        if nargin < 2
            defaultempty = [];
        end
        fieldunion = cellfun(@fieldnames, structures, 'UniformOutput', false);
        fieldunion = unique(vertcat(fieldunion{:}));
        structarray = repmat({defaultempty}, numel(structures), numel(fieldunion));
        for sidx = 1:numel(structures)
            [~, destcol] = ismember(fieldnames(structures{sidx}), fieldunion);
            structarray(sidx, destcol) = struct2cell(structures{sidx});
        end
        structarray = reshape(cell2struct(structarray, fieldunion, 2), size(structures));
    end
end
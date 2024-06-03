function generateEmbeddings(varargin)
    addpath('layers');
    addpath('utility');
    clc;
    
    if nargin < 1
        error("Pass at least 1 model ID");
    end
    
    for m=1:nargin
        runID = varargin{m};
        fprintf("RunID: %s\n", runID);
        tic;
            
        % load model 
        base_path = sprintf("model_dumps/%s", runID);
        model_filename = sprintf("%s/model.mat", base_path);
        config_filename = sprintf("%s/config.mat", base_path);
        config = load(config_filename);
        config = config.config; 
        
        model = load(model_filename);
        netE = model.netE;
        netD = model.netD;
        [~, ~, XTest, YTest] = loadDataset(config.dataset);
        

        % Batch Approach
        batchDimension = length(config.inputSize) + 1;
        dsTestX = arrayDatastore(XTest, IterationDimension=batchDimension);
        dsTestY = arrayDatastore(YTest, IterationDimension=batchDimension);
        dsTest = combine(dsTestX, dsTestY); % Combine XTest and YTest

        numOutputs = 2;
        mbq = minibatchqueue(dsTest, numOutputs, ...
            MiniBatchSize = config.batchSize, ...
            MiniBatchFcn=@(dataX, dataY) deal(cat(batchDimension, dataX{:}), cat(batchDimension, dataY{:})), ...
            MiniBatchFormat=config.batchFormat);
        
        shuffle(mbq);
        
        embeddings = struct;

        layerName = 'fullyConnectedLayer_3';
        layerIndex = find(arrayfun(@(l) strcmp(l.Name, layerName), netE.Layers));
        layersToKeep = netE.Layers(1:layerIndex);
        truncatedNet = dlnetwork(layersToKeep);
    
        while hasdata(mbq)
    
            [X, YTrue] = next(mbq);
            Z = forward(truncatedNet, X);
            
            for i=1:size(YTrue, 1)
                label = YTrue(i);
                l = sprintf('%d', label);
                disp(class(l));
                if isfield(l, embeddings)
                    embeddings.(l) = embeddings.(l) + Z;
                else
                    embeddings.(l) = Z;
                end
            end
            
        end

        
        [counts, ls] = groupcounts(YTest);
        for i=1:numel(counts)
            name = sprintf("x%d", ls(i));
            embeddings.(name) = embeddings.(name) ./ counts(i);
        end
        elapsedTime = toc;
        fprintf("Completed in %.4fs\n", elapsedTime);


        embeddings_filename = sprintf("%s/embeddings.mat", base_path);
        save(embeddings_filename, "embeddings");
    end
end

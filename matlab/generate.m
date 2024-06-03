function generate(varargin)
    addpath('layers');
    addpath('utility');
    clc;
    
    if nargin < 1
        error("Pass at least 1 model ID");
    end
    
    for m=1:nargin
        runID = varargin{m};
        
        % init figure 
        fig = figure;
        tiledlayout(1, 10, 'TileSpacing', 'compact', 'Padding', 'compact');
        set(fig, 'Position', [200, 200, 800, 150]);
            
        % load model 
        base_path = sprintf("model_dumps/%s", runID);
        model_filename = sprintf("%s/model.mat", base_path);
        embeddings_filename = sprintf("%s/embeddings.mat", base_path);
        config_filename = sprintf("%s/config.mat", base_path);
        
        if ~isfile(embeddings_filename)
            printf("No embedding found for %s\n", runID);
            continue;
        end

        config = load(config_filename);
        config = config.config; 
        embeddings = load(embeddings_filename);
        embeddings = embeddings.embeddings;
        
        model = load(model_filename);
        netD = model.netD;
        samplingType = model.netE.Layers(end);

        netD = model.netD;
        inputSize = netD.Layers(1).InputSize;

        for i=1:10
            l = i-1;
            embedding = embeddings.(sprintf("x%d", l));
            sample = forward(samplingType, embedding);
            sample = dlarray(sample, "CB");

            generated = forward(netD, sample);
            nexttile; 
            % title(sprintf("%d", l));
            title("a");
            imagesc(extractdata(generated));
            axis image;
        end
        % img_path = sprintf('model_dumps/%s/generated_images.png', runID);
        % saveas(fig, img_path);
    end
end

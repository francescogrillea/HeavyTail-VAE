function avgLossTest = test(netE, netD, XTest, labels, config)

    runID = config.runID;
    batchDimension = length(config.inputSize) + 1;
    dsTest = arrayDatastore(XTest, IterationDimension=batchDimension);
    numOutputs = 1;
    mbq = minibatchqueue(dsTest, numOutputs, ...
        MiniBatchSize = config.batchSize, ...
        MiniBatchFcn=@(dataX) cat(batchDimension, dataX{:}), ...
        MiniBatchFormat=config.batchFormat);
    
    shuffle(mbq);
    lossHistoryTest = [];

    while hasdata(mbq)

        X = next(mbq);
        Z = forward(netE, X);
        Y = forward(netD, Z);
       
        loss = mse(Y, X);
        lossHistoryTest = [lossHistoryTest, loss];
    end
    
    filename = sprintf('model_dumps/%s/lossTest.mat', runID);
    save(filename, "lossHistoryTest");
    avgLossTest = mean(lossHistoryTest);
    fprintf("Loss on Test Set: %.4f\n", avgLossTest);

    if strcmp(config.dataset, "mnist") | strcmp(config.dataset, "mnist_exp")
    
        chosenLabels_idx = [];
        for i=0:9
            all_i = find(labels == i);
            idx = randperm(length(all_i), 1);
            chosenLabels_idx = [chosenLabels_idx, all_i(idx)];
        end
        
        XTest = XTest(:, :, :, chosenLabels_idx);
        
        sizeXTest = size(XTest);
        numTestImages = size(XTest, 4);
        
        fig = figure;
        tiledlayout(2, numTestImages, 'TileSpacing', 'compact', 'Padding', 'compact');
        set(fig, 'Position', [200, 200, 1000, 300]);
        
        XTest = dlarray(XTest, "SSCB");
        for i = 1:numTestImages
            image = XTest(:,:,:,i);
            if strcmp(config.dataset, "mnist_exp")
                nexttile; imagesc(extractdata(log(image)));
                axis image;
            else
                nexttile; imagesc(extractdata(image));
                axis image;
            end
            if i == 1
                title('Original Images');
            end
        end
        for i = 1:numTestImages
            image = XTest(:,:,:,i);
            reconstructed = forward(netD, forward(netE, image));
            if strcmp(config.dataset, "mnist_exp")
                nexttile; imagesc(extractdata(log(reconstructed)));
                axis image;
            else
                nexttile; imagesc(extractdata(reconstructed));
                axis image;
            end
            if i == 1
                title('Reconstructed Images');
            end
        end
        img_path = sprintf('model_dumps/%s/reconstructed_images.png', runID);
        saveas(fig, img_path);

        if isfield(config, "noise") && config.noise ~= false
            fig = figure;
            tiledlayout(2, numTestImages, 'TileSpacing', 'compact', 'Padding', 'compact');
            set(fig, 'Position', [200, 200, 1000, 300]);
            
            for i = 1:numTestImages
                image = XTest(:,:,:,i);
                
                if strcmp(config.noise, "normal")
                    image = image + randn(size(image));
                elseif strcmp(config.noise, "logNormal")
                    image = image + lognrnd(size(image));
                elseif strcmp(config.noise, "uniform")
                    image = image + rand(size(image));
                end

                if strcmp(config.dataset, "mnist_exp")
                    nexttile; imagesc(extractdata(log(image)));
                    axis image;
                else
                    nexttile; imagesc(extractdata(image));
                    axis image;
                end
                if i == 1
                    title('Original Images');
                end
            end
            for i = 1:numTestImages
                image = XTest(:,:,:,i);
                reconstructed = forward(netD, forward(netE, image));
                if strcmp(config.dataset, "mnist_exp")
                    nexttile; imagesc(extractdata(log(reconstructed)));
                    axis image;
                else
                    nexttile; imagesc(extractdata(reconstructed));
                    axis image;
                end
                if i == 1
                    title('Reconstructed Images');
                end
            end
            img_path = sprintf('model_dumps/%s/denoised_images.png', runID);
            saveas(fig, img_path);
        end
    end
end

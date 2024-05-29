function avgLossTest = test(netE, netD, XTest, labels, config)

    runID = config.timestamp;
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

    if strcmp(config.dataset, "mnist")
    
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
        tiledlayout(numTestImages, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        XTest = dlarray(XTest, "SSCB");
        for i = 1:numTestImages
            image = XTest(:,:,:,i);
            % exp_image = exp(image);
            % reconstructed = forward(netD, forward(netE, exp_image));
            reconstructed = forward(netD, forward(netE, image));
            % reconstructed = log(reconstructed);
        
            nexttile; imagesc(extractdata(image));
            nexttile; imagesc(extractdata(reconstructed));
        end
        img_path = sprintf('model_dumps/%s/reconstructed_images.png', runID);
        saveas(fig, img_path);
    end

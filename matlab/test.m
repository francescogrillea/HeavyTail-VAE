function test(netE, netD, XTest, labels, runID)
    
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
    
        nexttile; imshow(extractdata(image));
        nexttile; imshow(extractdata(reconstructed));
    end

    img_path = sprintf('model_dumps/%s/reconstructed_images.png', runID);
    saveas(fig, img_path);

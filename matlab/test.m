function test(netE, netD, numLatentChannels, XTest)
% Generate a batch of new images by passing randomly sampled image encodings through the decoder.
numImages = 64;

samplingLayer = netE.Layers(end);
samplingFunction = @(X) samplingLayer.predict("", X);

ZNew = randn(numLatentChannels*2, numImages);
ZNew = samplingFunction(ZNew);
ZNew = dlarray(ZNew, "CB");

YNew = predict(netD, ZNew);
YNew = extractdata(YNew);

% Display the generated images in a figure.
figure
I = imtile(YNew);
imshow(I)
title("Generated Images");


% Compare test images with their reconstruction
numTestImages = size(XTest, 4);

figure
tiledlayout(numTestImages, 2);

XTest = dlarray(XTest, "SSCB");
for i = 1:numTestImages
    image = XTest(:,:,:,i);
    reconstructed = forward(netD, forward(netE, image));

    nexttile; imshow(extractdata(image));
    nexttile; imshow(extractdata(reconstructed));
end
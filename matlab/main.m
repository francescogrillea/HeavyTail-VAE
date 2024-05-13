% Config
config = struct;
config.numLatentChannels = 10;
config.neuronsPerLayer = 300;
config.hiddenLayersEncoder = 3;
config.hiddenLayersDecoder = 4;
config.sampleDistribution = "LogNormal";


% Load Dataset
dataset = load('mnist.mat');

XTrain = reshape(dataset.training.images, 28,28,1,[]);
XTest = reshape(dataset.test.images, 28,28,1,[]);

sizeXTrain = size(XTrain);
imageSize = sizeXTrain(1:end-1);

% Build Model
[netE, netD] = buildModel(imageSize, config);

% Train Model
[netE, netD] = train(netE, netD, XTrain, config);

% Test Model
test(netE, netD, config.numLatentChannels, XTest(:,:,:,1:5));
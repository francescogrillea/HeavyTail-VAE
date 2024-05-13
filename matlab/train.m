function [netE, netD] = train(netE, netD, XTrain, config)

DEFAULT_NUMEPOCHS = 3;
DEFAULT_BATCHSIZE = 32;
DEFAULT_LEARNINGRATE = 1e-3;

if nargin < 4
    config = struct;
end
if ~isfield(config, "numEpochs")
    config.numEpochs = DEFAULT_NUMEPOCHS;
end
if ~isfield(config, "learningRate")
    config.learningRate = DEFAULT_LEARNINGRATE;
end
if ~isfield(config, "batchSize")
    config.batchSize = DEFAULT_BATCHSIZE;
end
if ~isfield(config, "plotLoss")
    config.plotLoss = false;
end

if config.plotLoss
    % Calculate the total number of iterations for the training progress monitor
    numObservationsTrain = size(XTrain,4);
    numIterationsPerEpoch = ceil(numObservationsTrain / config.batchSize);
    numIterations = config.numEpochs * numIterationsPerEpoch;

    % Initialize the training progress monitor
    monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info="Epoch", ...
    XLabel="Iteration");
else
    monitor.Stop = false;
end


% Create a minibatchqueue object that processes and manages mini-batches of images during training
dsTrain = arrayDatastore(XTrain, IterationDimension=4);
numOutputs = 1;

mbq = minibatchqueue(dsTrain, numOutputs, ...
    MiniBatchSize = config.batchSize, ...
    MiniBatchFcn=@(dataX) cat(4, dataX{:}), ...
    MiniBatchFormat="SSCB");


% Initialize the parameters for the Adam solver.
trailingAvgE = [];
trailingAvgSqE = [];
trailingAvgD = [];
trailingAvgSqD = [];


epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < config.numEpochs && ~monitor.Stop
    tic
    epoch = epoch + 1;
    fprintf("Epoch %d\n", epoch);

    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        X = next(mbq);

        % Evaluate loss and gradients.
        [loss,gradientsE,gradientsD] = dlfeval(@modelLoss,netE,netD,X);

        % Update learnable parameters.
        [netE,trailingAvgE,trailingAvgSqE] = adamupdate(netE, ...
            gradientsE, trailingAvgE, trailingAvgSqE, iteration, config.learningRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD, trailingAvgD, trailingAvgSqD, iteration, config.learningRate);

        if config.plotLoss
            % Update the training progress monitor.
            recordMetrics(monitor, iteration, Loss=loss);
            updateInfo(monitor, Epoch=epoch + " of " + config.numEpochs);
            monitor.Progress = 100*iteration/numIterations;
        end
    end

    toc
end


function [loss, gradientsE, gradientsD] = modelLoss(netE, netD, X)
    % Forward through encoder.
    [Z, mu, logSigmaSq] = forward(netE, X);
    
    % Forward through decoder.
    Y = forward(netD, Z);
    
    % Calculate loss and gradients.
    reconstructionLoss = mse(Y, X);
    
    KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq), 1);
    KL = mean(KL);
    
    loss = reconstructionLoss + KL;
    
    [gradientsE,gradientsD] = dlgradient(loss, netE.Learnables, netD.Learnables);
end

end
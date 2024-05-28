function [trainingSet, trainingLabels, testSet, testLabels] = loadDataset(dataset)

    if strcmp(dataset, 'exp_mnist')
        datasetStruct = load("datasets/mnist.mat");
        trainingSet = exp(reshape(datasetStruct.training.images, 28, 28, 1, []));
        trainingLabels = datasetStruct.training.labels;
        testSet = exp(reshape(datasetStruct.test.images, 28, 28, 1, []));
        testLabels = datasetStruct.test.labels;

    elseif strcmp(dataset, 'mnist')
        datasetStruct = load(sprintf("datasets/%s.mat", dataset));
        trainingSet = reshape(datasetStruct.training.images, 28, 28, 1, []);
        trainingLabels = datasetStruct.training.labels;
        testSet = reshape(datasetStruct.test.images, 28, 28, 1, []);
        testLabels = datasetStruct.test.labels;

    elseif strcmp(dataset, 'lognorm') | strcmp(dataset, "overlapping_lognorm")
        datasetStruct = load(sprintf("datasets/%s.mat", dataset));
        trainingSet = datasetStruct.out(:,1:2)';
        trainingLabels = datasetStruct.out(:, 3);
        % testSet = 
        % trainingLabels =
    else
        error("Dataset not supported")
    end
end
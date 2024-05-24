addpath("layers")

clc; clear;

dataset = load('mnist.mat');
XTest = reshape(dataset.test.images, 28,28,1,[]);
labels = dataset.test.labels;

runID = "2024-05-24_10-07-07";
filepath = sprintf("model_dumps/%s/model.mat", runID);

model = load(filepath);
netE = model.netE;
netD = model.netD;

test(netE, netD, XTest, labels, runID);
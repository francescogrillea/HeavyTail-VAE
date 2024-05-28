function run_test(varargin)
addpath("layers")
addpath("utility")

for i = 1:nargin
    runID = varargin{i};
    filepath = sprintf("model_dumps/%s/model.mat", runID);
    model = load(filepath);
    
    [~, ~, XTest, YTest] = loadDataset(model.config.dataset);
    
    netE = model.netE;
    netD = model.netD;
    
    test(netE, netD, XTest, YTest, runID);
end
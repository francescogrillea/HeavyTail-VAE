function run_test(varargin)
addpath("layers")
addpath("utility")

for i = 1:nargin
    runID = varargin{i};

    base_path = sprintf("model_dumps/%s", runID);
    model_filename = sprintf("%s/model.mat", base_path);
    config_filename = sprintf("%s/config.mat", base_path);

    model = load(model_filename);
    netE = model.netE;
    netD = model.netD;

    config = load(config_filename);
    config = config.config; 
    [~, ~, XTest, YTest] = loadDataset(config.dataset);
       
    test(netE, netD, XTest, YTest, config);
end
clear; clc;

normalize = true;

mu = [0, 2;
    2, 0];
sigma = [0.4, 0.4;
    0.4, 0.4];

n_training = 20000;
n_test = 5000;
n = n_training + n_test;

pd = @(mu, sigma)lognrnd(mu, sigma, n, 1);

d1 = generate_samples(pd, mu(1,:), sigma(1,:));
d1 = [d1, zeros(n, 1)];

d2 = generate_samples(pd, mu(2,:), sigma(2,:));
d2 = [d2, ones(n, 1)];

if normalize
    d = [d1; d2];
    
    a = rescale(d(:, 1));
    b = rescale(d(:, 2));
    d(:, 1) = a;
    d(:, 2) = b;
    
    idx = (d(:, end) == 0);
    d1 = d(idx, :);
    d2 = d(~idx, :);
end


d1_train = d1(1:n_training, :);
d1_test = d1(n_training+1:n, :);

d2_train = d2(1:n_training, :);
d2_test = d2(n_training+1:n, :);


fig = figure;
scatter(d1(:, 1), d1(:, 2), "filled", "r");
hold on
scatter(d2(:, 1), d2(:, 2), "filled", "b");

training = [d1_train; d2_train];
idx = randperm(2*n_training);
training_struct = struct;
training_struct.counts = size(training, 1);
training_perm = training(idx, :);
training_struct.instances = training_perm(:, 1:2)';
training_struct.labels = training_perm(:, 3);

test = [d1_test; d2_test];
idx = randperm(2*n_test);
test_struct = struct;
test_struct.counts = size(test, 1);
test_perm = test(idx, :);
test_struct.instances = test_perm(:, 1:2)';
test_struct.labels = test_perm(:, 3);

training = training_struct;
test = test_struct;

dataset_basepath = "datasets";
dataset_filename = "lognorm_rescaled";
save(sprintf("%s/%s.mat", dataset_basepath, dataset_filename), "training", "test");
saveas(fig, sprintf("%s/%s.png", dataset_basepath, dataset_filename));


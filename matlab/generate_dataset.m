clear; clc;

mu = [0, 1;
    1, 0];
sigma = [0.4, 0.4;
    0.4, 0.4];

n = 20000;


pd = @(mu, sigma)lognrnd(mu, sigma, n, 1);
d1 = generate_samples(pd, mu(1,:), sigma(1,:));

d1 = [d1, zeros(n, 1)];
scatter(d1(:, 1), d1(:, 2), "filled", "r");
hold on

d2 = generate_samples(pd, mu(2,:), sigma(2,:));
d2 = [d2, ones(n, 1)];

scatter(d2(:, 1), d2(:, 2), "filled", "b");

out = [d1; d2];

indexes = randperm(2*n);
out = out(indexes, :);
dataset_filepath = sprintf("datasets/lognorm.mat");
save(dataset_filepath, "out");


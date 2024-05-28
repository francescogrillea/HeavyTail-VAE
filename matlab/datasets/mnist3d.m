clear; clc;

n_new_channels = 2;
ds = load('C:\Projects\HeavyTailed-VAE\matlab\datasets\mnist.mat');

mnist = ds.training.images;
n = size(mnist, 3);

mnist = reshape(mnist, 28, 28, 1, n);
for i=1:n_new_channels
    channel = randn(28, 28, 1, n);
    channel = normalize(channel, "range");
    mnist = cat(3, mnist, channel);
end
ds.training.images = mnist;
training = ds.training;

mnist = ds.test.images;
n = size(mnist, 3);

mnist = reshape(mnist, 28, 28, 1, n);
for i=1:n_new_channels
    channel = randn(28, 28, 1, n);
    channel = normalize(channel, "range");
    mnist = cat(3, mnist, channel);
end
ds.test.images = mnist;
test = ds.test;

save("mnist3d.mat", "training", "test");

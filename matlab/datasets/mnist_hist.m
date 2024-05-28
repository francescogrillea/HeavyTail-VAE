dataset = load('mnist.mat');
dataset = dataset.training.images;

pixel_values = dataset(:);

histogram(pixel_values, 20);
title('Histogram of Pixel Values in MNIST Dataset');
xlabel('Pixel Value');
ylabel('Frequency');



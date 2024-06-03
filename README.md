# HeavyTailed-VAE

Matlab implementation of a Variational Auto-Encoder using the following distributions to learn a latent space
* Normal
* LogNormal
* Inverse Gaussian

# Installation
```bash
git clone https://github.com/francescogrillea/HeavyTailed-VAE
cd HeavyTailed-VAE/matlab
mkdir embeddings model_dumps
```

# Usage
The main script will create a new model and train it from scratch, according to the configuration specified in `config.json` file, and then evaluate the reconstruction error on the test set. 
If you want to run some configuration example, copy the desired file from `matlab/config_examples/` to `matlab` folder. 
For each configuration file, a folder will be created in `model_dumps/` and named with its runID, generated as `[dataset]-[samplingLayerType]-[timestamp]`.

```matlab
> main("config.json")
```
More configurations can be executed in a single run, simply `main("config1.json", "config2.json")`.

To run test separately, execute:
```matlab
> run_test("runID")
```
where `runID` is the folder name in `model_dumps/` for a specific run.

To use the model as a generative model, you need first to generate data embeddings for each class, then can be used to generate new samples. 
```matlab
> generateEmbeddings("runID")
> generate("runID")
```
Models and results are stored in `model_dumps/runID`.

# Results
Read report for more details.

| runID                                                   | dataset | noise    | samplingLayer                | numLatentChannels                                | numEncoderLayers                               | numDecoderLayers                             | numEpochs                                  | learningRate                           | batchSize                       | kl                          | avgEpochTime           | finalLoss      | testLoss  | 
|---------------------------------------------------------|-------|----------|------------------------------|--------------------------------------------------|------------------------------------------------|----------------------------------------------|--------------------------------------------|----------------------------------------|---------------------------------|-----------------------------|------------------------|----------------|-----------|
| mnist-normalSamplingLayer-2024-06-02_16-06-03           | mnist | None     | normalSamplingLayer          | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 8.6414                 | 6.0516         | 5.6380    | 
| mnist-logNormalSamplingLayer-2024-06-02_16-20-37        | mnist | None     | logNormalSamplingLayer       | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 5.2543                 | 5.9924         | 6.0242    |
| mnist-inverseGaussianSamplingLayer-2024-06-02_16-29-29  | mnist | None     | inverseGaussianSamplingLayer | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.3547                 | 5.9608         | 6.3093    |
| mnist-normalSamplingLayer-2024-06-02_18-09-17           | mnist | normal   | normalSamplingLayer          | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.7505                 | 11.2881        | 7.9923    |
| mnist-logNormalSamplingLayer-2024-06-02_18-22-19        | mnist | normal   | logNormalSamplingLayer       | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.3903                 | 10.3644        | 8.1758    |
| mnist-inverseGaussianSamplingLayer-2024-06-02_18-36-17  | mnist | normal   | inverseGaussianSamplingLayer | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.8947                 | 10.9453        | 9.1947    |
| mnist-normalSamplingLayer-2024-06-02_18-44-14           | mnist	| logNormal |	normalSamplingLayer	| 10	                                              | 3                                              | 	4	                                          | 100                                        | 	0.0003                                | 	512                            | 	0.01                       | 	5.217	                | 6.6633	        | 25.2696	  |
| mnist-logNormalSamplingLayer-2024-06-02_18-53-00        | mnist |	logNormal |	logNormalSamplingLayer | 10                                               | 	3                                             | 	4                                           | 100                                        | 	0.0003                                | 	512                            | 	0.01                       | 	6.6629                | 	6.0478        | 	29.9268	 |
| mnist-inverseGaussianSamplingLayer-2024-06-02_19-04-12  | mnist | 	logNormal |	inverseGaussianSamplingLayer	| 10| 	3| 	4| 	100| 	0.0003| 	512| 	0.01| 	5.6769	| 6.4028| 	24.5316  |	 
| mnist-normalSamplingLayer-2024-06-02_19-06-57           | mnist | uniform  | normalSamplingLayer          | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.5532                 | 9.8695         | 8.3692    | NaN          |
| mnist-logNormalSamplingLayer-2024-06-02_19-15-39        | mnist | uniform  | logNormalSamplingLayer       | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 7.7549                 | 9.6213         | 9.2726    | NaN          |
| mnist-inverseGaussianSamplingLayer-2024-06-02_19-30-47  | mnist | uniform  | inverseGaussianSamplingLayer | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 5.8202                 | 6.8273         | 8.7814    | NaN          |
| lognorm-normalSamplingLayer-2024-06-02_20-41-55         | lognorm | None     | normalSamplingLayer          | 2                                                | 3                                              | 4                                            | 50                                         | 0.0003                                 | 100                             | 0.01                        | 5.8159                 | 0.094373       | 0.012862  | NaN          |
| lognorm-logNormalSamplingLayer-2024-06-02_20-46-47      | lognorm | None     | logNormalSamplingLayer       | 2                                                | 3                                              | 4                                            | 50                                         | 0.0003                                 | 100                             | 0.01                        | 5.166                  | 0.13061        | 0.032739  | NaN          |
| lognorm-inverseGaussianSamplingLayer-2024-06-02_20-51-07 | lognorm | None     | inverseGaussianSamplingLayer | 2                                                | 3                                              | 4                                            | 50                                         | 0.0003                                 | 100                             | 0.01                        | 5.9583                 | 0.2218         | 0.1945    | NaN          |
| mnist_exp-normalSamplingLayer-2024-06-02_21-20-58       | mnist_exp | None     | normalSamplingLayer          | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 8.6677                 | 18.9327        | 18.9708   | NaN          |
| mnist_exp-logNormalSamplingLayer-2024-06-02_21-35-30    | mnist_exp | None     | logNormalSamplingLayer       | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 5.8203                 | 20.3062        | 21.029    | NaN          |
| mnist_exp-inverseGaussianSamplingLayer-2024-06-02_21-45-17 | mnist_exp | None     | inverseGaussianSamplingLayer | 10                                               | 3                                              | 4                                            | 100                                        | 0.0003                                 | 512                             | 0.01                        | 6.7812                 | 21.5641        | 22.1914   | NaN          |

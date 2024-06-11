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
Read [report](2024_SEAI_project_C05__grillea_canzoneri_.pdf) for more details.

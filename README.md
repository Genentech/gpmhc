This is the repo for performing inference using the baseline graph-pmhc model introduced in https://www.biorxiv.org/content/10.1101/2023.01.19.524779v1

# Inference Instructions
First clone the repo and enter a terminal in the repo folder
Next, create your virtual environment: 
```
mamba env create -f environment.yml
```
[!NOTE] Conda was not properly installing libraries and dependencies, so please use mamba (I used mamba 1.4.1) You will likely have to choose different torch/dgl packages to support your version of CUDA, I used 11.8. If you do so, delete the existing versions from environment.yml and install them according to the instructions on their website after creating the environment. To ensure that the code functions properly, use the same versions of DGL/torch used here (1.1.0 and 1.13.1). Sorry, this library doesn't support cpu inference.

Then, download the model file (link to be created) and put it in the models/baseline_model directory.
Next, download the CSV you'd like to get inference on (link to be created) or supply your own (ensure that the csv formatting matches the template)

You are now ready for inference. In terminal launch the infer file:
```
python ./gpmhc/infer.py
```
If you'd like to perform inference on a different file than the template add a keyword:
```
python ./gpmhc/infer.py --csv 'path/to/file.csv'
```
Keywords for using different model parameters and model architectures are not currently supported.

# Fun Facts
The model_json is the way that most arguments/settings/hyperparameters are handled in this repo, as we are only supporting inference, these arguments are not be be changed, and will be set by the model_json provided. Regardless, you may be curious about what the different keys in the model_json provided correspond to.

number of epochs
```
"epochs": 30, 
```
Random seed
```
"seed": 2906,
``` 
This is a list of adjacency matrices, one for DR, DP, and DQ, each sublist represents a position on the binding core, and each index represents the residue location in the psuedosequence
```
"mhc_adj": [[[0, 4, 5, 6, 7, 8, 9, 10, 38, 39, 40, 41, 42], [4, 10, 36, 37, 38, 39], [1, 3, 4, 10, 11, 12, 13, 37, 39], [1, 4, 13, 23, 24, 25, 34, 35, 37], [13, 33, 34], [2, 13, 14, 15, 17, 22, 23, 26, 34], [14, 17, 25, 26, 28, 31, 32, 34], [14, 16, 17, 30, 31], [17, 18, 19, 20, 21, 27, 29, 30, 31]], [[0, 3, 4, 5, 6, 7, 8, 9, 39, 40, 41, 42], [0, 9, 37, 38, 39, 40], [0, 2, 9, 10, 11, 12, 38, 40], [0, 12, 23, 24, 25, 34, 35, 36, 38], [12, 13, 23, 25, 34, 35], [1, 12, 13, 14, 16, 21, 22, 23, 25, 26, 27, 35], [13, 16, 25, 27, 29, 32, 33, 35], [13, 15, 16, 31, 32, 33], [13, 16, 17, 18, 19, 20, 28, 30, 31, 32]], [[4, 5, 6, 7, 8, 29, 30, 31], [0, 4, 7, 8, 27, 28, 29, 30], [0, 1, 3, 4, 8, 10, 27, 28, 29, 30], [0, 1, 2, 3, 10, 17, 18, 19, 26, 28], [9, 10, 17, 25, 26], [2, 9, 10, 11, 13, 16, 17, 20], [11, 12, 13, 20, 23, 24, 25], [11, 12, 13, 22, 23], [12, 13, 14, 15, 21, 22, 23]]], 
```
This is the length of the psuedosequences for each gene
```
"mhc_lens": [43, 43, 32],
```
The loss options are used for different loss functions
```
"loss_options": {"loss_func": "MaskedBCEWithLogitsLoss"}
```
Dataloader options contains several subset options (this would make more sense in the context of the training code)
```
"dataloader_options": {"csv_to_df":
``` 
Schema options gives the indices from the full MHC chain that are used in the psuedosequence for each gene, the max psuedosequence length, whether or not netmhcpan's psuedosequence is used (it isn't self consistent and so required some hard coding), and the amount of padding. Currently, netmhcpan's psuedosequence and padding are not supported in this library
```
{"option": "schema_options": [[6, 8, 10, 21, 23, 30, 31, 42, 51, 52, 53, 57, 58, 61, 64, 65, 67, 68, 71, 72, 75], [8, 10, 12, 25, 27, 29, 36, 46, 56, 59, 60, 66, 69, 70, 73, 76, 77, 80, 81, 84, 85, 88], [8, 10, 21, 23, 30, 31, 42, 51, 52, 53, 57, 58, 61, 64, 65, 67, 68, 71, 72, 75], [8, 10, 11, 12, 23, 25, 26, 27, 34, 44, 54, 57, 58, 64, 67, 68, 71, 74, 75, 78, 79, 82, 83], [10, 11, 13, 24, 26, 34, 54, 55, 56, 63, 64, 67, 70, 71, 74, 78], [11, 13, 15, 30, 32, 59, 62, 63, 69, 72, 76, 79, 80, 83, 84, 87], 43, "normal", 0]}},
```
model_hyper_opts define the model hyperparameters which are used while generating the model, the following were used in the baseline model.
```
"model_hyper_opts": {"batch_size": 64, "lr": 0.0005, "node_feat_size": 64, "graph_feat_size": 128, "edge_feat_size": 3, "gnn_layers": 2, "gnn_dropout": 0.1, "timesteps": 2, "rnn_dropout": 0.2, "classifier_dropout": 0.4, "bc_pad": 0, "readout": "recurrent", "posenc": 1, "time_steps": 2}}

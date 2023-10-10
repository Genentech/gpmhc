import json
import pandas as pd
import numpy as np
import fire
import text_extensions_for_pandas as tp
import gpmhc.data as data_utils
import gpmhc.learner as learner_utils
import os

#Only one GPU inference is currently suppported, cpu inference not supported
def infer(input_model='./models/baseline_model/', sub_model='model_final',csv='./data/mhc2_small_df.csv', arch='baseline_model'):
    model_path = os.getcwd() + input_model[1:] + sub_model
    '''
     So using the getcwd function here will only work if they run the script from the gpmhc directory, if for some reason they start the script from another directory, get cwd will return that directory.
     Usually what i do is ,
     SCRIPT_LOCATION = os.path.dirname(os.path.realpath(__file__))
     This returns the location of the current file, not the directory where the user started python from.  From here you can navigate to the correct location for the model directory, and is less likely to break.
     
    '''
    dataset = csv.split('/')[-1].split('.')[0]

    #get parameters for generating learner
    json_input = json.load(open(input_model + "json_input.json", "r"))
    #flag for inference mode, set model arch
    json_input['inference'] = 1
    json_input['arch'] = arch
    json_input['model_dir'] = input_model
    #generate learner
    learner = learner_utils.json_to_learner(json_input)
    #load weights
    learner = learner.load(model_path)

    #read input csv and cleanup the csv
    df = pd.read_csv(csv)
    df = data_utils.cleanup_schema(df,json_input['dataloader_options']['csv_to_df']['schema_options'])
    
    #Move model to cuda
    learner.model = learner.model.cuda()
    '''
        Do you expect the users to always run on GPU/Cuda?
    '''
    
    #Generate dataloader
    test_dl = learner.arch.get_dataloaders_test(df)
    
    #Do inference
    with learner.parallel_ctx():  y_pred, y_true = y_pred, y_true = learner.get_preds(dl=test_dl, reorder=False)
    '''
        I am not sure what happens if you use this parallel context on a single GPU, do you expect users to always have atleast 2 GPUs?  You may want to add some options to adjust the batch size so they can run it on smaller GPU's if they want.
    '''
    #split output into binding core start location (graph_idx) and predictions (y_pred), over all alleles (16)
    graph_idx = y_pred[:,16:].cpu().detach().numpy()
    y_pred = y_pred[:,:16]
    y_pred = np.array(y_pred.cpu().detach().numpy())

    #Get max prediction, which is the one representative of the datum
    y_pred_max = np.max(y_pred, axis=1).flatten()
    y_true = np.array(y_true.cpu().detach().numpy())
    #Put all predictions of all alleles in csv
    learner.arch.df_test['EL_pred_all'] = tp.TensorArray(y_pred)

    #Save all predictions, target values, and selected binding core start location for each allele/peptide
    np.save(learner.arch.json_input['model_dir']+f'/{dataset}_y_pred.npy', y_pred)
    np.save(learner.arch.json_input['model_dir']+f'/{dataset}_y_true.npy', y_true)
    np.save(learner.arch.json_input['model_dir']+f'/{dataset}_graph_idx.npy', graph_idx)

    #ELution_pred is the model prediction
    learner.arch.df_test['EL_pred'] = y_pred_max
    #get predicted binding core for each peptide using best allele/binding core start location
    arg_max = np.argmax(y_pred,axis=1)
    cores = np.array([pep[int(g[a]):int(g[a])+9] for g,a,pep in zip(graph_idx,arg_max, learner.arch.df_test['peptide'].values)])
    learner.arch.df_test['peptide_core'] = cores
    #save csv
    learner.arch.df_test.to_csv(learner.arch.json_input['model_dir']+f'/{dataset}.csv')
    return None


if __name__ == "__main__":
    fire.Fire(infer)
from functools import partial
from fastai.distributed import *
import importlib
import json
from fastai.learner import Learner
from fastai.optimizer import Adam
import torch
import accelerate
from pathlib import Path


#use the parameter json to create the learner
def json_to_learner(json_input):
    set_seed(json_input['seed'])
    model_hyper_opts, arch, dataloader_options, loss_options = parse_json(json_input)
    model_arch = importlib.import_module(f"gpmhc.{arch}").model(json_input=json_input)
    learner = model_arch.get_learner(model_dir=Path(json_input['model_dir']))
    learner.json_input = json_input
    return learner

#set/fix random seed
def set_seed(seed):
    accelerate.utils.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#parse the parameter json
def parse_json(json_input):
    if type(json_input) == str:
        json_input = json.loads(json_input)
    return json_input['model_hyper_opts'], json_input['arch'], json_input['dataloader_options'], json_input['loss_options']


#instantiates the fastai learner from parameter json
def get_learner(model):
    data = None

    opt_func = partial(Adam, lr=model.arch.json_input['model_hyper_opts']['lr'], wd=0.00)
    loss_func = model.arch.loss_func

    learner = Learner(
        data,
        model,
        #path=model.arch.json_input['model_dir'],
        opt_func=opt_func,
        loss_func=loss_func,
    )

    #so you can later call the init information from model (from the architecture file you are using)
    learner.arch = model.arch
    return learner
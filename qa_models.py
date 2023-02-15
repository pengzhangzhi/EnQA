import re
import subprocess
import os
import ray
import pandas as pd
import numpy as np
import torch
import esm
from biopandas.pdb import PandasPdb
from network.resEGNN import resEGNN_with_ne
import importlib  
EnQA = importlib.import_module("EnQA-MSA")


ray.init()

def mergePDB(inputPDB, outputPDB, newStart=1):
    with open(inputPDB, 'r') as f:
        x = f.readlines()
    filtered = [i for i in x if re.match(r'^ATOM.+', i)]
    chains = set([i[21] for i in x if re.match(r'^ATOM.+', i)])
    chains = list(chains)
    chains.sort()
    with open(outputPDB + '.tmp', 'w') as f:
        f.writelines(filtered)
    merge_cmd = 'pdb_selchain -{} {} | pdb_chain -A | pdb_reres -{} > {}'.format(','.join(chains),
                                                                                 outputPDB + '.tmp',
                                                                                 newStart,
                                                                                 outputPDB)
    subprocess.run(args=merge_cmd, shell=True)
    os.remove(outputPDB + '.tmp')
    
    
def merge_all_pdbs(directory):
   """
   Merges all pdb files in a directory.
   """
   for root, dirs, files in os.walk(directory):
      for file in files:
         if file.endswith(".pdb"):
            path = os.path.join(root,file)
            out_path = os.path.join(root,file.split('.')[0] + '_merged.pdb')
            print("Merging %s" % out_path)
            mergePDB(path,out_path)
            

def qa_casp15(directory,out_dir):
    """
    Gets the casp15 case name, model name, model path, and output path.
    Args:
        directory: path to directory containing casp15 models.
            example:
            -casp15_models/
                -T0959/
                    -T0959_1.pdb
        out_dir: path to directory to save results.
    Returns:
        the casp15 case name, model name, model path, and output path.
    """
    os.makedirs(name=out_dir, exist_ok=True)
    futures = []
    for case in os.listdir(directory):
        case_dir = os.path.join(directory,case)
        if not os.path.isdir(os.path.join(directory,case)):
            continue
        case_out_dir = os.path.join(out_dir,case)
        os.makedirs(case_out_dir, exist_ok=True)
        
        future = qa_on_a_case.remote(case_dir, case_out_dir,)
        futures.append(future)
    ray.get(futures)
    
@ray.remote(num_gpus=1)
def qa_on_a_case(case_dir, case_out_dir):
    csv_path = os.path.join(case_out_dir, 'qa.csv')
    if os.path.exists(csv_path):
        return
    device = "cuda"
    # load esm model
    model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm.eval()

    # load qa model    
    dim1d = 24
    dim2d = 145
    model = resEGNN_with_ne(dim2d=dim2d, dim1d=dim1d)
    model.to(device)
    model.load_state_dict(torch.load("models/EnQA-MSA.pth", map_location=device))
    model.eval()
    
    score_df = pd.DataFrame(columns=['model_name', 'lddt'])
    for model_file in os.listdir(case_dir):
        if model_file.endswith("merged.pdb"):
            model_name = model_file.split('.')[0]
            model_path = os.path.join(case_dir,model_file)
            out_path = os.path.join(case_out_dir,model_name)
            os.makedirs(out_path, exist_ok=True)
            try:
                lddts = EnQA.run(model_path, out_path, model_esm, model,alphabet)
                mean_lddt = lddts.mean()
                score_df = score_df.append({'model_name': model_name, 'lddt': float(mean_lddt)}, ignore_index=True)
            except Exception as e:
                with open(os.path.join('error.txt'), 'a') as f:
                    f.write(str(e) + '\t'+model_path + '\n')
    
    score_df.to_csv(csv_path, index=False)



qa_casp15("casp15_data","casp15_qa_results")
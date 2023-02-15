import os

import torch
import argparse
import numpy as np
import esm
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb

from data.loader import expand_sh
from feature import create_basic_features
from network.resEGNN import resEGNN_with_ne

@torch.no_grad()
def run(input,output, model_esm, model,alphabet):
    out_path = os.path.join(output, os.path.basename(input).replace('.pdb', '.npy'))
    
    if os.path.exists(out_path):
        return np.load(out_path)
    
    if not os.path.isdir(output):
        os.mkdir(output)
    device = next(model.parameters()).device
    
    one_hot, features, pos_data, sh_adj, el = create_basic_features(input, output)
    
    # plddt
    ppdb = PandasPdb()
    ppdb.read_pdb(input)
    plddt = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']['b_factor']
    plddt = plddt.to_numpy().astype(np.float32) / 100

    
    parser = PDBParser(QUIET=True)
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    
    
    
    structure = parser.get_structure('struct', input)
    for m in structure:
        for chain in m:
            seq = []
            for residue in chain:
                seq.append(d3to1[residue.resname])
    
    # Load ESM-2 model
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", ''.join(seq))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[6], return_contacts=True)
    
    token_representations = results["attentions"].numpy()
    token_representations = np.reshape(token_representations, (-1, len(seq) + 2, len(seq) + 2))
    token_representations = token_representations[:, 1: len(seq) + 1, 1: len(seq) + 1]
    
    
    
    pred_lddt = []
    af2_plddt = []
    true_lddt = []
    
    x = [one_hot, features, np.expand_dims(plddt, axis=0)]
    f1d = torch.tensor(np.concatenate(x, 0)).to(device)
    f1d = torch.unsqueeze(f1d, 0)
    
    x2d = [expand_sh(sh_adj, f1d.shape[2]), token_representations]
    f2d = torch.tensor(np.concatenate(x2d, 0)).to(device)
    f2d = torch.unsqueeze(f2d, 0)
    pos = torch.tensor(pos_data).to(device)
    dmap = cdist(pos_data, pos_data)
    el = np.where(dmap <= 0.15)
    cmap = dmap <= 0.15
    cmap = torch.tensor(cmap.astype(np.float32)).to(device)
    el = [torch.tensor(i).to(device) for i in el]
    with torch.no_grad():
        _, _, lddt_pred = model(f1d, f2d, pos, el, cmap)
    
    out = lddt_pred.cpu().detach().numpy().astype(np.float16)
    
    np.save(out_path, out)        
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pdb file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')

    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
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
    
    out_plddt = run(args.input, args.output, model_esm, model,alphabet)


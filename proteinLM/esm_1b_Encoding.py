import torch
import esm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

'''
1. Main Models

ESM-1b (UR50): *esm1b_t33_650M_UR50S()*

ESM-MSA-1b (UR50 + MSA): *esm_msa1b_t12_100M_UR50S()*

ESM-1v (UR90): *esm1v_t33_650M_UR90S_\[1-5\]()*

ESM-IF1 (CATH + UR50): *esm_if1_gvp4_t16_142M_UR50()*


Note that here we use ESM-1b model. May try others later.
'''
# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


'''
2. Load protein dataset

csv file, cdhit clusters (all=curated+derived)
'''
PATH_allInfo = '/home/qinqin/dimeng/idrs/pycode/data/all_cdhit_info.csv'
df_allInfo = pd.read_csv(PATH_allInfo)


'''
3. Encoding
3.1. Processing data
    Bug - ValueError: Sequence length 34352 above maximum sequence length of 1024
    
    solution - keep seq which len<=1024
        19076 out of 20625, ~92.5% sequence left

3.2. Train & Test Split
    80% Training
    20% Test

3.3. Saving
'''
# 3.1. Processing Data
df_info1024 = df_allInfo[df_allInfo['protein_length']<=1024]

# 3.2. Train & Test split
df_train, df_test = train_test_split(df_info1024, test_size=0.2)

# 3.3. Saving

def saveEncoding(df_info, fpath):
    '''
    Aim:
        Encoding sequence first, then save it to file fpath.
    params:
        df_info: Dataframe, contains protein_name, protein_length, annotation and sequence at least!
        fpath: str, path to save encoding dataset. same form as one-hot encoding.
    '''
    with open(fpath, 'w') as f:
        f.write(str(df_info.shape[0]))
        f.write('\n')
        f.write('1280 2')
        f.write('\n')
        for index, row in df_info.iterrows():
            
            ###
            # 1. Encoding
            ###
            data = [(row['protein_name'], row['sequence'])]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            token_representations = token_representations.reshape(-1).tolist()
            
            ###
            # 2. saving
            ###
            f.write(row['protein_name'])
            f.write('\n')
            # f.write(row['annotation_method'])
            # f.write('\n')
            f.write(str(row['protein_length']))
            f.write('\n')
            # f.write(''.join(map(str, row['encoded_seq']))[2:-2])
            f.write(' '.join(str(x) for x in token_representations))
            # f.write(''.join(map(str, row['encoded_seq'][0])))
            f.write('\n')
            # f.write(seq_flatten(row['annotation']))
            f.write(" ".join(row['annotation']))
            f.write('\n')
            f.write('\n')
        f.close()
            
# Train & Test file paths
fpath_train = '/home/qinqin/dimeng/idrs/pycode/data/esm_1b_train.txt'
fpath_test = '/home/qinqin/dimeng/idrs/pycode/data/esm_1b_test.txt'

# Saving
saveEncoding(df_train, fpath_train)
saveEncoding(df_test, fpath_test)
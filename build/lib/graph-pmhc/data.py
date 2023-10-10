import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def cleanup_schema(df,schema_options, infer_all=False):
    #These are the names of the MHC sequence columns
    allele_seq_cols = ['mhc_dr1_1', 'mhc_dr1_2', 'mhc_dr3_1', 'mhc_dr3_2', 'mhc_dr4_1', 'mhc_dr4_2', 'mhc_dr5_1', 'mhc_dr5_2', 'mhc_dp1_1', 'mhc_dp1_2', 'mhc_dp1_3', 'mhc_dp1_4','mhc_dq1_1', 'mhc_dq1_2', 'mhc_dq1_3', 'mhc_dq1_4']
    #Derive the psuedosequence from the input indices
    df = get_psuedos(df, allele_seq_cols,schema_options)
    # * is the pad token, pad sequences to the correct length if too short
    df[allele_seq_cols] = df[allele_seq_cols].fillna('*' * schema_options[6])

    df = df.reset_index(drop=True)
    #$ is the empty flank symbol
    df['nFlank'] = df['nFlank'].fillna('$')
    df['cFlank'] = df['cFlank'].fillna('$')
    df['peptide'] = df['peptide'].fillna('*')

    
    if not 'EL' in df:
        df['EL']=0
    if not 'split' in df:
        df['split']='test'
    df = df.dropna(subset=["EL"]).reset_index(drop=True)
    
    if not 'cFlank' in df:
        df['cFlank'] = '$'
    
    #Pad is an unused option used if one chooses to use a binding core greater than length 9
    pad = schema_options[8]
    df[f'peptide'] = '['*pad + df['peptide'] + '['*pad
    df['peptide_length'] = df['peptide'].str.len()    
    return df


def tokenize(df, tokenizer_config):
    total_size = 0
    data = None
    #this pads and concatenates the columns requested by the tokenizer config into one long input
    for key, val in tokenizer_config.items():
        if data is None: data = df[key].str.pad(val['size'], side=val['side'], fillchar=val['token'])
        else: data = data + df[key].str.pad(val['size'], side=val['side'], fillchar=val['token'])
        total_size = total_size + val['size']
    #Sets up characters so they can be converted to numbers quickly
    data = data.values.astype(f'S{total_size}').view(np.dtype(('S1',total_size)))
    # Numericalize
    data = fast_cat(data)
    # Handle special cases, X an dL token, shouldn't be used ideally
    x_val = categories_val[np.where(categories == 'X')[0][0]]
    l_val = categories_val[np.where(categories == 'L')[0][0]]
    
    data[data==x_val] = l_val
    return data

def fast_cat(a):
    # converts char bytes to integers fast, places '*' at 0
    raw = (a.view(np.ubyte) - 96).astype('int32') - 192 - 10 - 22
    raw[raw == -22] = 0  # fix so * goes to zero
    raw[raw == 22] = 2 # The next three map residues to lower numbers so 1-20 is densely packed
    raw[raw == 23] = 10
    raw[raw == 25] = 15
    raw[raw == -28] = 21  # fix so $ goes to 2
    raw[raw == -31] = 22  # fix so '!' goes to 10
    raw[raw == 31] = 0  # turns _ a special token
    
    return raw


categories = np.array(['*', 'A', 'R', 'N', 'D', 'C', 'Q', 'E',
                       'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                       'T', 'W', 'Y', 'V', '$', '!', 'X', 'U', '_', '[', ']'])

categories_val = fast_cat(categories.astype('S1'))
# Make sure no overlap in categories/tokens
assert np.alltrue(np.unique(fast_cat(categories.astype('S1')), return_counts=True)[1])


def get_psuedos(df, allele_seq_cols,schema_options=None):
    seq_df = pd.read_csv('./graph-pmhc/mhc_seq_df.csv')
    #See dataframe for allotype formatting, codebase requires this formatting
    allos = df['allotype'].unique()
    #creates alpha/beta pairs from input
    full_seqs = np.array([get_full_sequences(allo, seq_df) for allo in df['allotype'].unique()])
    
    #These are the indices used in the graph
    DRA = np.array(schema_options[0])
    DRB = np.array(schema_options[1])
    DPA = np.array(schema_options[2])
    DPB = np.array(schema_options[3])
    DQA = np.array(schema_options[4])
    DQB = np.array(schema_options[5])
    
    #IMGT aligns all alleles so the starting point of the B chain is fixed
    p_start = 229
    r_start = 229
    q_start = 232

    #This grabs the psudeosequence elements and orders the alleles properly. Netmhcpan has some inconsistences in how it defines psuedosequences for certain alleles, this is handled here if netmhcpan's psuedosequence is used
    for i in range(16):
        if i<8:
            for j in range(len(full_seqs)):
                if full_seqs[j,i]==full_seqs[j,i]:
                    full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DRA].tolist()+np.array(list(full_seqs[j,i]))[DRB+r_start].tolist())
        if 7<i<12:
            for j in range(len(full_seqs)):
                if full_seqs[j,i]==full_seqs[j,i]:
                    full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DPA].tolist()+np.array(list(full_seqs[j,i]))[DPB+p_start].tolist())
        if 11<i<16:
            for j, allo in enumerate(allos):
                if full_seqs[j,i]==full_seqs[j,i]:
                    if 'netmhcpan' in schema_options:
                        if allo[5:10]=='02:01':
                            DQA_0201 = np.array([10,13,24,26,33,52,54,60,61,63,67,68,70,74,75])
                            full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DQA_0201].tolist()+np.array(list(full_seqs[j,i]))[DQB+q_start].tolist())
                        elif allo[5:10]=='05:05' or allo[5:10]=='05:01' or allo[5:10]=='04:01':
                            DQA_0505 = np.array([10,13,24,26,33,54,59,60,61,63,67,68,70,74,75])
                            full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DQA_0505].tolist()+np.array(list(full_seqs[j,i]))[DQB+q_start].tolist())
                        else:
                            full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DQA].tolist()+np.array(list(full_seqs[j,i]))[DQB+q_start].tolist())
                    else: full_seqs[j,i] = ''.join(np.array(list(full_seqs[j,i]))[DQA].tolist()+np.array(list(full_seqs[j,i]))[DQB+q_start].tolist())

    #put psuedos in dict for fast mapping in pandas
    seq_dicts = [{allo:seq for allo,seq in zip(allos, f_seq)} for f_seq in full_seqs.transpose()]
    
    for col,di in zip(allele_seq_cols, seq_dicts):
        df[col] = df['allotype'].map(di)
    return df

def get_full_sequences(allotype, df):
    #each allele is separated with ____
    allotype = np.array(allotype.split('___'))

    #Split up DR,DQ,DP
    DR = allotype[[allo[1]=='R' for allo in allotype]]
    DQ = allotype[[allo[1]=='Q' for allo in allotype]]
    DP = allotype[[allo[1]=='P' for allo in allotype]]

    #all DR alleles begin with DRA*01:01
    DR_temp = ['IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLPSTEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGVRKSNAAERRGPL' + ''.join(np.array(list(df[df['allele']==allele]['full_sequence'].values[0]))) for allele in DR[1:]]
    DR = np.repeat(np.nan, 8).astype('object')
    for i,d in enumerate(DR_temp):
        DR[i] = d
    
    #Generate all possible alpha and beta pairs
    DQ_A = DQ[[allo[2]=='A' for allo in DQ]]
    DQ_B = DQ[[allo[2]=='B' for allo in DQ]]
    DQ = np.repeat(np.nan, 4).astype('object')

    #grab the sequences from the dataframe for each allele, and combine each alpha with each beta
    counter = 0
    for a in DQ_A:
        for b in DQ_B:
            temp = ''.join(np.array(list(df[df['allele']==a]['full_sequence'].values[0]))) + ''.join(np.array(list(df[df['allele']==b]['full_sequence'].values[0])))
            DQ[counter] = temp
            counter+=1

    DP_A = DP[[allo[2]=='A' for allo in DP]]
    DP_B = DP[[allo[2]=='B' for allo in DP]]
    DP = np.repeat(np.nan, 4).astype('object')
    counter = 0
    for a in DP_A:
        for b in DP_B:
            temp = ''.join(np.array(list(df[df['allele']==a]['full_sequence'].values[0]))) + ''.join(np.array(list(df[df['allele']==b]['full_sequence'].values[0])))
            DP[counter] = temp
            counter+=1

    #get a list of alleles
    allotype = np.array(np.concatenate((DR,DP,DQ)))
    return allotype

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        return x, y
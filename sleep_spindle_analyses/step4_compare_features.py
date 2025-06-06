import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from collections import defaultdict
from tqdm import tqdm
from pingouin import partial_corr


df = pd.read_csv('features_spindles.csv.zip')

df2_asd = pd.read_excel('ASD-TD(egi-data).xlsx', sheet_name='ASD')
df2_td = pd.read_excel('ASD-TD(egi-data).xlsx', sheet_name='TD')
cols = ['病案号', '性别', '年龄']
df2 = pd.concat([df2_asd[cols], df2_td[cols]], axis=0, ignore_index=True)
df2 = df2.rename(columns={'病案号':'SID', '性别':'Sex', '年龄':'Age'})
df2['Sex'] = df2['Sex']-1
df2['SexTXT'] = ''
df2.loc[df2.Sex==0, 'SexTXT'] = 'M'
df2.loc[df2.Sex==1, 'SexTXT'] = 'F'
df2['SID'] = df2.SID.astype(str)
df['SID'] = df.SID.astype(str)

ids_df = []
ids_df2 = []
for i in range(len(df2)):
    matched_idx = np.where(df.SID.str.contains(df2.SID.iloc[i]))[0]
    if len(matched_idx)==1:
        ids_df.append(matched_idx[0])
        ids_df2.append(i)
    elif len(matched_idx)>1:
        raise Exception('?')
df = pd.concat([
    df.iloc[ids_df].reset_index(drop=True),
    df2.loc[ids_df2, ['Age','Sex','SexTXT']].reset_index(drop=True) ], axis=1)
df['Group2'] = 0
df.loc[df.Group=='TD', 'Group2'] = 0
df.loc[df.Group=='ASD', 'Group2'] = 1

cols = [x for x in df.columns if 'SP_DENS' in x] + \
    [x for x in df.columns if 'SP_CDENS' in x] + \
    [x for x in df.columns if 'SP_AMP' in x] + \
    [x for x in df.columns if 'SP_ISA_S' in x] + \
    [x for x in df.columns if 'SP_DUR' in x] + \
    [x for x in df.columns if 'SP_NOSC' in x] + \
    [x for x in df.columns if 'SP_FFT' in x] + \
    [x for x in df.columns if 'SP_CHIRP' in x] + \
    [x for x in df.columns if 'SO_DUR' in x] + \
    [x for x in df.columns if 'SO_P2P' in x] + \
    [x for x in df.columns if 'SO_RATE' in x] + \
    [x for x in df.columns if 'SO_SLOPE' in x] + \
    [x for x in df.columns if 'SP_COUPL_OVERLAP' in x]
    
    
for suffix in ['all', 'male', 'female']:
    if suffix=='all':
        df_ = df.copy()
    elif suffix=='male':
        df_ = df[df.SexTXT=='M'].reset_index(drop=True)
    elif suffix=='female':
        df_ = df[df.SexTXT=='F'].reset_index(drop=True)
        
    df_res = defaultdict(list)
    for col in tqdm(cols):
        asd = df_.loc[df_.Group=='ASD',col].values
        td  = df_.loc[df_.Group=='TD',col].values
        asd = asd[~np.isnan(asd)]
        td = td[~np.isnan(td)]
        
        if len(asd)>=5 and len(td)>=5:
            df_res['Name'].append(col)
            df_res['Median(TD)'].append( np.median(td) )
            df_res['Median(ASD)'].append( np.median(asd) )
            df_res['Mean(TD)'].append( np.mean(td) )
            df_res['Mean(ASD)'].append( np.mean(asd) )
            df_res['Cohen_d'].append( (np.mean(asd)-np.mean(td))/np.std(np.r_[td,asd]) )
            df_res['P(unadjusted)'].append( mannwhitneyu(asd,td).pvalue )
            
            if suffix in ['male', 'female']:
                covar = ['Age']
            else:
                covar = ['Age', 'Sex']
            res = partial_corr(df_, x='Group2', y=col, covar=covar, method='spearman')
            df_res['P(adjusted)'].append( res['p-val'].iloc[0] )
            
            df_res['N(TD)'].append( len(td) )
            df_res['N(ASD)'].append( len(asd) )

    df_res = pd.DataFrame(df_res)
    df_res['Significant'] = (df_res['P(adjusted)']<0.05/13).astype(int)
    df_res = df_res.sort_values('P(adjusted)')
    print(suffix)
    print(df_res)
    df_res.to_excel(f'comparison_results_{suffix}.xlsx', index=False)

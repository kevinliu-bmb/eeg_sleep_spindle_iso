import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pingouin import partial_corr


df = pd.read_csv('features_spindles.csv.zip')

df2 = pd.read_excel('ASD-TD(egi-data).xlsx', sheet_name='ASD')
df2 = df2.rename(columns={'病案号':'SID', '性别':'Sex', '年龄':'Age', '精细\n动作':'精细动作', '个人\n社交':'个人社交'})
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
    df2.loc[ids_df2].drop(columns='SID').reset_index(drop=True) ], axis=1)

cols1 = [x for x in df.columns if 'SP_DENS' in x] + \
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
cols2_primary = ['ADOS总分(2019-9-11始）', 'CARS']
cols2_secondary = ['适应性', '大运动', '精细动作', '语言', '个人社交', '沟通', '学前功能', '自我管理', '休闲', '社交', '社区应用', '居家生活', '健康安全', '自我照顾', '动作技巧', '概念技能', '社会技能', '实用技能', '一般适应综合', '沟通', '互动']
cols2 = cols2_primary + cols2_secondary
import pdb;pdb.set_trace()
for suffix in ['all', 'male', 'female']:
    if suffix=='all':
        df_ = df.copy()
    elif suffix=='male':
        df_ = df[df.SexTXT=='M'].reset_index(drop=True)
    elif suffix=='female':
        df_ = df[df.SexTXT=='F'].reset_index(drop=True)
        
    df_res = []
    for col1 in tqdm(cols1):
        for col2 in cols2:
            if suffix in ['male', 'female']:
                covar = ['Age']
            else:
                covar = ['Age', 'Sex']
            res = partial_corr(df_, x=col1, y=col2, covar=covar, method='pearson')
            res.insert(0, 'EEG', col1)
            res.insert(0, 'ASDParameter', col2)
            df_res.append(res)

    df_res = pd.concat(df_res, axis=0, ignore_index=True)
    df_res = df_res[['ASDParameter', 'EEG', 'r', 'CI95%', 'p-val', 'n']]
    df_res = df_res.sort_values('p-val')
    df_res1 = df_res[np.in1d(df_res.ASDParameter, cols2_primary)]
    df_res1['Significant'] = (df_res1['p-val']<0.05/13/len(cols2_primary)).astype(int)
    df_res2 = df_res[np.in1d(df_res.ASDParameter, cols2_secondary)]
    print(suffix)
    print(df_res1)
    print(df_res2)
    with pd.ExcelWriter(f'corr_results_{suffix}_pearsonR.xlsx') as writer:
        df_res1.to_excel(writer, sheet_name='Primary', index=False)
        df_res2.to_excel(writer, sheet_name='Secondary', index=False)
    

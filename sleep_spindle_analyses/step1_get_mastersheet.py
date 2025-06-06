import os, glob
import pandas as pd
#from pyedflib.highlevel import read_edf_header
import mne
from tqdm import tqdm


td_paths = glob.glob('/data/haoqisun/Autistm-sleep-Kong/asdtdegidata-edf/TD/*.edf')
asd_paths = glob.glob('/data/haoqisun/Autistm-sleep-Kong/asdtdegidata-edf/ASD/*.edf')
td_paths = [x for x in td_paths if '4231776_20210706_141717_fil_1' not in x]

df = pd.DataFrame(data={
    'Group':['TD']*len(td_paths) + ['ASD']*len(asd_paths),
    'EDFPath':td_paths+asd_paths})
df.insert(0, 'SID', df.EDFPath.str.split('/',expand=True)[6].str.replace('.edf',''))
df['SleepStagePath'] = '/data/haoqisun/Autistm-sleep-Kong/sleep_stages2/'+df.Group+'/sleep_stages_'+df.SID+'.npz'
df['ArtifactPath'] = '/data/haoqisun/Autistm-sleep-Kong/artifacts/'+df.Group+'/artifacts_'+df.SID+'.npz'

for i in range(len(df)):
    if not os.path.exists(df.SleepStagePath.iloc[i]):
        path = df.SleepStagePath.iloc[i].replace('sleep_stages2', 'sleep_stages')
        assert os.path.exists(path), path
        df.loc[i, 'SleepStagePath'] = path
        
df['Channels'] = ''
df['Fs'] = 0.
for i in tqdm(range(len(df))):
    sid = df.SID.iloc[i]
    #hdr = read_edf_header(df.EDFPath.iloc[i])
    #df.loc[i,'Channels'] = str(hdr[''])
    edf = mne.io.read_raw_edf(df.EDFPath.iloc[i], verbose=False, preload=False)
    df.loc[i,'Channels'] = str(edf.ch_names)
    df.loc[i,'Fs'] = edf.info['sfreq']
    if len(edf.ch_names)>33:
        df.loc[i, 'EDFPath'] = f'/data/haoqisun/Autistm-sleep-Kong/edf-64to32/{sid}.edf'

df.to_excel('mastersheet.xlsx', index=False)


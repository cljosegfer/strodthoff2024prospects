
import argparse
import wfdb
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import resampy
import datetime
# import h5pickle as h5py
import h5py

from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from reproducao.reproducao_timeseries_utils import dataset_add_mean_col, dataset_add_std_col, dataset_add_length_col, dataset_get_stats, save_dataset
from reproducao.reproducao_ecg_utils import get_stratified_kfolds, resample_data, fix_nans_and_clip

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}

def main():
    parser = argparse.ArgumentParser(description='A script to extract two paths from the command line.')
    
    # Add arguments for the two paths
    # parser.add_argument('--data-path', help='path to mimic ecg subset', default='D:\datasets\MIMIC-IV-ECG-DEMO\mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0')
    parser.add_argument('--data-path', help='path to mimic ecg subset', default='/home_cerberus/disk2/luizfacury/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0')
    # parser.add_argument('--target-path', help='desired output path', default='D:\datasets\MIMIC-IV-ECG-DEMO\mimic-strodthoff')
    parser.add_argument('--target-path', help='desired output path', default='/home_cerberus/disk2/josefernandes/mimic-strodthoff')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    data_path = args.data_path
    clip_amp=3
    target_fs=100
    channels=12
    strat_folds=20
    channel_stoi=channel_stoi_default
    target_folder = args.target_path
    recreate_data=True
    
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    print('writing in', target_folder)
    
    print('walking in data path', data_path)
    lst = []
    for root, dirs, files in tqdm(os.walk(data_path)):
        relative = os.path.relpath(root, data_path)
        for file in files:
            lst.append(os.path.join(relative, file))
    lst = [x for x in lst if x.endswith(".hea")]
    
    size = len(lst)
    length = 1000
    leads = 12
    print('writing h5 file with {} samples'.format(size))
    output_hdf5 = h5py.File(os.path.join(target_folder, 'MIMICstrodthoff.h5'), 'w')
    x = output_hdf5.create_dataset('tracings', shape = (size, length, leads), dtype = np.float32)
    study_id = output_hdf5.create_dataset('study_id', shape = (size,), dtype = np.int64)
    subject_id = output_hdf5.create_dataset('subject_id', shape = (size,), dtype = np.int64)
    
    print('populating h5 file with {} samples'.format(size))
    meta = []
    for idx, l in tqdm(enumerate(lst)):
        # archive.extract(l, path="tmp_dir/")
        # archive.extract(l[:-3]+"dat", path="tmp_dir/")
        filename = Path(data_path)/l
        sigbufs, header = wfdb.rdsamp(str(filename)[:-4])

        tmp={}
        tmp["data"]=filename.parent.parent.stem+"_"+filename.parent.stem # subject_id + patientid_study
        tmp["study_id"]=int(filename.stem)
        tmp["subject_id"]=int(filename.parent.parent.stem[1:])
        tmp['ecg_time']= datetime.datetime.combine(header["base_date"],header["base_time"])
        tmp["nans"]= list(np.sum(np.isnan(sigbufs),axis=0))#save nans channel-dependent
        if(np.sum(tmp["nans"])>0):#fix nans
            fix_nans_and_clip(sigbufs,clip_amp=clip_amp)
        elif(clip_amp>0):
            sigbufs = np.clip(sigbufs,a_max=clip_amp,a_min=-clip_amp)

        data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
        
        assert(target_fs<=header['fs'])
        meta.append(tmp)
        x[idx, :, :] = data
        study_id[idx] = tmp['study_id']
        subject_id[idx] = tmp['subject_id']
    
    print('writing pandas dataframe')
    df = pd.DataFrame(meta)
    dataset_add_mean_col(df,output_hdf5, data_folder=target_folder)
    dataset_add_std_col(df,output_hdf5, data_folder=target_folder)
    dataset_add_length_col(df,output_hdf5, data_folder=target_folder)

    #save means and stds
    mean, std = dataset_get_stats(df)
    #save
    lbl_itos=[]
    save_dataset(df,lbl_itos,mean,std,target_folder)
    
    print('closing h5 file')
    output_hdf5.close()

if __name__ == '__main__':
    main()

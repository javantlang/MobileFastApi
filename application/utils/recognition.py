import numpy as np
import pandas as pd
import wfdb
import glob
import joblib
from scipy import signal
from biosppy.signals import ecg


class Recognition:

    @staticmethod
    def get_data(file):
        normal_sinus_rhythm = joblib.load("./utils/joblibs/normal_sinus_rhythm.joblib")
        arrhythmia = joblib.load("./utils/joblibs/arrhythmia.joblib")
        atrial_fibrillation = joblib.load("./utils/joblibs/atrial_fibrillation.joblib")
        malignant_ventricular_ectopy = joblib.load("./utils/joblibs/malignant_ventricular_ectopy.joblib")
        supraventricular_arrhythmia = joblib.load("./utils/joblibs/supraventricular_arrhythmia.joblib")

        signal_list=[]
        for i in range(len(file)):
            record = wfdb.rdrecord(file[:-4]) 
            fs=record.__dict__['fs']
            channel_number=record.__dict__['n_sig']

            for j in range(channel_number):
                signal_=record.__dict__['p_signal'][:, j][0:100000]
                out = ecg.ecg(signal=signal_, sampling_rate=fs, show=False)
                out_array=out['templates']
            
                for k in range(out_array.shape[0]):
                    signal_list.append(out_array[k])
            
        rescaled_list=[]
        for i in range(len(signal_list)):
            rescaled_list.append(signal.resample(signal_list[i],76))
            
        X=np.array(rescaled_list)
        
        normal_sinus_rhythm=np.mean(normal_sinus_rhythm.predict(X), axis = None)
        arrhythmia=np.mean(arrhythmia.predict(X), axis = None)
        atrial_fibrillation=np.mean(atrial_fibrillation.predict(X), axis = None) 
        malignant_ventricular_ectopy=np.mean(malignant_ventricular_ectopy.predict(X), axis = None) 
        supraventricular_arrhythmia=np.mean(supraventricular_arrhythmia.predict(X), axis = None) 
        
        file_name = file.split('\\')[-1]
        column_list=[file_name,
                    round(normal_sinus_rhythm,2),
                    round(atrial_fibrillation,2),
                    round(malignant_ventricular_ectopy,2),
                    round(supraventricular_arrhythmia,2)]
        
        COL_NAMES=[ 'file name',
                    'normal sinus rhythm',
                    'atrial fibrillation',
                    'malignant ventricular ectopy',
                    'supraventricular arrhythmia']

        df = pd.DataFrame(np.column_stack(column_list), columns=COL_NAMES)    
        return df
    
    @staticmethod
    def recognize(unzip_folder):
        folder = rf'{unzip_folder}/*.hea'
        files = glob.glob(folder)

        data_list=[]
        for file in files:
            data_list.append(Recognition.get_data(file))
            
        df = pd.concat(data_list, axis=0, ignore_index=True)

        df.reset_index().to_csv('result.csv')
        return df

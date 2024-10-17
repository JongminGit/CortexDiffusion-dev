import os
import shutil
fold1 = ['MMRR-21-14.pt', 'HLN-12-2.pt', 'MMRR-21-19.pt', 'HLN-12-6.pt', 'NKI-TRT-20-1.pt', 'MMRR-21-5.pt', 'MMRR-21-17.pt', 'MMRR-21-4.pt', 'OASIS-TRT-20-14.pt', 'NKI-RS-22-1.pt', 'OASIS-TRT-20-5.pt', 'OASIS-TRT-20-12.pt', 'NKI-RS-22-15.pt', 'Twins-2-2.pt', 'OASIS-TRT-20-17.pt', 'MMRR-21-3.pt', 'MMRR-21-9.pt', 'OASIS-TRT-20-19.pt', 'NKI-RS-22-16.pt', 'MMRR-21-1.pt']
fold2 = ['OASIS-TRT-20-15.pt', 'NKI-TRT-20-2.pt', 'HLN-12-11.pt', 'MMRR-21-15.pt', 'MMRR-21-10.pt', 'MMRR-21-16.pt', 'MMRR-21-6.pt', 'NKI-TRT-20-7.pt', 'NKI-RS-22-8.pt', 'NKI-TRT-20-17.pt', 'NKI-TRT-20-4.pt', 'NKI-RS-22-21.pt', 'NKI-TRT-20-19.pt', 'MMRR-21-2.pt', 'Twins-2-1.pt', 'NKI-RS-22-18.pt', 'OASIS-TRT-20-8.pt', 'Afterthought-1.pt', 'NKI-TRT-20-3.pt', 'MMRR-3T7T-2-1.pt']
fold3 = ['NKI-RS-22-9.pt', 'OASIS-TRT-20-4.pt', 'HLN-12-10.pt', 'OASIS-TRT-20-7.pt', 'NKI-RS-22-7.pt', 'OASIS-TRT-20-20.pt', 'MMRR-21-20.pt', 'NKI-RS-22-12.pt', 'NKI-RS-22-6.pt', 'OASIS-TRT-20-2.pt', 'NKI-TRT-20-6.pt', 'NKI-TRT-20-20.pt', 'NKI-TRT-20-10.pt', 'NKI-RS-22-14.pt', 'HLN-12-5.pt', 'OASIS-TRT-20-13.pt', 'NKI-RS-22-11.pt', 'MMRR-21-13.pt', 'NKI-TRT-20-8.pt', 'NKI-TRT-20-18.pt']
fold4 = ['MMRR-21-7.pt', 'OASIS-TRT-20-10.pt', 'HLN-12-8.pt', 'OASIS-TRT-20-16.pt', 'MMRR-3T7T-2-2.pt', 'MMRR-21-12.pt', 'NKI-TRT-20-14.pt', 'MMRR-21-21.pt', 'NKI-TRT-20-5.pt', 'NKI-RS-22-17.pt', 'NKI-TRT-20-11.pt', 'NKI-RS-22-10.pt', 'NKI-RS-22-22.pt', 'NKI-TRT-20-12.pt', 'NKI-TRT-20-15.pt', 'MMRR-21-8.pt', 'HLN-12-12.pt', 'NKI-RS-22-4.pt', 'OASIS-TRT-20-9.pt', 'NKI-RS-22-20.pt']
fold5 = ['MMRR-21-18.pt', 'HLN-12-4.pt', 'HLN-12-7.pt', 'HLN-12-3.pt', 'NKI-RS-22-2.pt', 'HLN-12-9.pt', 'Colin27-1.pt', 'OASIS-TRT-20-3.pt', 'OASIS-TRT-20-6.pt', 'NKI-RS-22-5.pt', 'MMRR-21-11.pt', 'NKI-TRT-20-13.pt', 'NKI-RS-22-3.pt', 'NKI-RS-22-13.pt', 'OASIS-TRT-20-11.pt', 'HLN-12-1.pt', 'NKI-RS-22-19.pt', 'OASIS-TRT-20-18.pt', 'NKI-TRT-20-9.pt', 'NKI-TRT-20-16.pt']

data_path = "/data/lfs/kimjongmin8/Develop/CortexDiffusion/Mindboggle_dataset"
data_list = os.listdir(data_path)

for data in data_list:
    if data in fold1:
        shutil.move(os.path.join(data_path,data), os.path.join(data_path,"white/fold1"))
    
    if data in fold2:
        shutil.move(os.path.join(data_path,data), os.path.join(data_path,"white/fold2"))
    
    if data in fold3:
        shutil.move(os.path.join(data_path,data), os.path.join(data_path,"white/fold3"))
    
    if data in fold4:
        shutil.move(os.path.join(data_path,data), os.path.join(data_path,"white/fold4"))
    
    if data in fold5:
        shutil.move(os.path.join(data_path,data), os.path.join(data_path,"white/fold5"))
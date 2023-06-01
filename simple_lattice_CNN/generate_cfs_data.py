import re
import os
import sys
import shutil
from shutil import copyfile, copy2
from shutil import move

import numpy as np
import pandas as pd

from latt2D_modules import calc_diffuse_cfs4, calc_diffuse_cfs4_big,recalc_diff_from_other_cfs4_big
from latt2D_modules import get_occ_map, get_2D_occ_map_from_seq,store_occ_map_as_seq
from latt2D_modules import plot_occ_map,read_bin,output_16bit_pgm
import time
import h5py

def init_isingMC_infiles(cfs=4,corrin=False,jsw=False):
    
    '''
    create files to init the ising inputs with zeros and jsw with ones 
    '''

    if not np.any(corrin): corrin=np.zeros((cfs,cfs)) 
    corrin[0,0]=1.0
    fhout=open('corr.in','w')
    for row in range(cfs):
        fhout.write(' '.join(["%.6f"%i for i in corrin[row]])+'\n')
    fhout.close()
        
        
    if not np.any(jsw): jsw=np.ones((cfs,cfs)) 
        
    fhout=open('jswitch.in','w')
    for row in range(cfs):
        fhout.write(' '.join(["%d"%i for i in jsw[row]])+'\n')
    fhout.close()




def save_as_h5(img_dir,h5name,nfiles=1,Nstart=0):

    # Set the path to your images directory
    # img_dir = "./cfs4_big_image_inputs_bin/"

    # Set the size of your images
    img_size = (64, 64)

    # Set the path to where you want to save the HDF5 file
    h5_file = "%s.h5"%(h5name)

    # Create a new HDF5 file
    with h5py.File(h5_file, 'w') as hf:

        # Create a dataset to store your image data
        img_data = hf.create_dataset(h5name, shape=(nfiles, *img_size), dtype=np.float64)

        # Loop through each image file
        for i in range(nfiles):    
        
            # Convert the image to a numpy array
            img_arr =read_bin(img_dir+'/hk0_%s.bin'%str(i+Nstart).zfill(6), npixels=64, offset=1280)
        
            # Store the image data in the dataset
            img_data[i] = img_arr

    # Print a message indicating the dataset has been created
    print("Dataset created in {}".format(h5_file))

    
    
def generate_cfs4_big_data(bin_dir,seq_dir,csv_out_name,exp=0,N=2,Nstart=0,iconc=0.50,cread=0,icycles=500,ianneal=500 ):

    start_time = time.time()
    
    if not os.path.exists(bin_dir):
        print ("%s did not exist! creating it ! " %(bin_dir) )
        os.makedirs(bin_dir)
        
    if not os.path.exists(seq_dir):
        print ("%s did not exist! creating it ! " %(seq_dir) )
        os.makedirs(seq_dir)

    
    # when possible always dump to a list or array before converting to Dataframe
    corrfuncs=[]
    for i in range(N):
        k=i+Nstart
        print("generating image %d at %.2f seconds"%(k,start_time-time.time()) )
        calc_diffuse_cfs4_big(iconc,cread,icycles,ianneal,exp)
        copyfile('./expfiles_%d/hk0.bin'%exp,'%s/hk0_%s.bin'% (bin_dir,str(k).zfill(6)) )
        store_occ_map_as_seq('./expfiles_%d/ising2D_occ.txt'%exp,'%s/ising2D_seq_%s.dat' %  (seq_dir, str(k).zfill(6) ))
        corr_out=np.loadtxt('expfiles_%d/corr.out'%exp)
        corrfuncs.append(corr_out.flatten())
        
    end_time = time.time()
    total_time = end_time - start_time

    print("Total time taken: {:.2f} seconds".format(total_time))
    
    df=pd.DataFrame(corrfuncs)
    df.columns=['00','01','02','03','10','11','12','13','20','21','22','23','30','31','32','33']
    df.to_csv(csv_out_name)
    df.head()
    
    save_as_h5(bin_dir,bin_dir,nfiles=N,Nstart=Nstart)
    
    
    
## to specify a desired input matrix use the below syntax   

# corrin=np.random.randn(4,4)
# jsw=np.random.randint(0,2,size=(4,4))
# init_isingMC_infiles(cfs=4,corrin=corrin,jsw=jsw)

# this will initalize with zeros
# init_isingMC_infiles(cfs=4)

# generate_cfs4_big_data('cfs4_big_bin_1','cfs4_big_seq_1','cfs4_big_corr_out_1.csv',exp=0,N=2,Nstart=0,iconc=0.50,cread=0,icycles=500,ianneal=500)


bin_dir = sys.argv[1]
seq_dir = sys.argv[2]
csv_out_name = sys.argv[3]
exp = sys.argv[4]
N = sys.argv[5]

generate_cfs4_big_data(bin_dir,seq_dir,csv_out_name,int(exp),int(N))
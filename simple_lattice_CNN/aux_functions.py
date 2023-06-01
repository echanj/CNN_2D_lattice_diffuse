
import re
import os
import sys
import shutil
from shutil import copyfile, copy2
from shutil import move
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from scipy import stats

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import h5py

from latt2D_modules import get_2D_occ_map_from_seq,read_bin
from latt2D_modules import get_occ_map

from matplotlib.animation import FuncAnimation

def model_evaluate_and_plot(model,history,X,y):

    fig, ax = plt.subplots(1, 3, figsize=(13,3))
    score = model.evaluate(X, y, verbose=1)
    print("Loss:", score[0])
    print("Mean Square Error:", score[1])
    print("Mean Absolute Error:", score[2])
    
    # Plot the training and validation loss as a function of the epoch
    ax[0].semilogy(history.history['loss'], label='Training loss')
    ax[0].semilogy(history.history['val_loss'], label='Validation loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Log(Loss)')
#    ax[0].set_ylim([0,2])
#    ax[0].margins(0.25)
    ax[0].set_title('Best MSE: %.4f , MAE: %.4f '% (score[1],score[2]),size=10 )
    ax[0].legend()

# Evaluate the model on the test set
    y_pred = model.predict(X)
    r2 = r2_score(y.flatten(), y_pred.flatten())
    print("Test R2 score:", r2)

    m,b=np.polyfit(y.flatten(),y_pred.flatten(),1)
    xs=np.linspace(-1,1,10)
    ys=m*xs+b
    
    sns.scatterplot(ax=ax[1],x=y.flatten(),y=y_pred.flatten(),marker='o',edgecolor='b',c='none',s=10,alpha=0.7,label='R2:%.2f'%r2)
    ax[1].plot(xs,ys,c='k',marker='none',linestyle='--',alpha=0.6,label='line fit')
    ax[1].set_xlim([-1.0,1.0])
    ax[1].set_ylim([-1.0,1.0])
    ax[1].margins(0.25)
    ax[1].set_xlabel('Actual Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].set_title('Predicted vs Actual Values',size=10)
    ax[1].legend()
    
    residuals = y.flatten() - y_pred.flatten()
    ax[2].scatter(y_pred.flatten(), residuals,marker='s',edgecolor='r',c='none',s=25,alpha=0.3,label='R2:%.2f'%r2)
    ax[2].axhline(0.0,linestyle='--',color='k')
    ax[2].set_xlabel('Predicted values')
    ax[2].set_ylabel('Residuals')
    ax[2].set_title('Residuals of Predicted Values: %s' %model.name,size=10)
    ax[2].legend()
    
    fig.subplots_adjust(wspace=0.4, hspace=0.2)

def get_CFS_data_anim(df,seq_dir='./image_inputs_seq',bin_dir='image_inputs_bin',cfs=2,nframes=200):
    
    

    # Define a function to initialize  
    def init_CFS_data_anim():
        idx=0
        k=df.image_idx.iloc[idx]
        myL2=df.L2.iloc[idx]
        # Update the data for the plot
        
        occ2d=get_2D_occ_map_from_seq('%s/ising2D_seq_%s.dat'%(seq_dir,str(k).zfill(6)))
        axes[0].imshow(np.transpose(occ2d),interpolation='nearest',cmap='gray') 
        axes[0].axis("off")
        imdat = read_bin('%s/hk0_%s.bin'%(bin_dir,str(k).zfill(6)), npixels=64, offset=1280)
        axes[1].imshow(np.flip(imdat,0),cmap='gray')
        axes[1].axis("off")
        corrf=df.iloc[idx][1:(cfs*cfs+1)].values
      
        sns.heatmap(corrf.reshape((cfs,-1)), ax=axes[2], cmap='bwr',square=True, annot=False ,cbar_kws={'shrink':0.75},vmax=1.0,vmin=-1.0  ) #cbar=0 )
  
        axes[2].axis('off')
        fig.suptitle("CFS%d vector L2 distance from origin %.3f " %(cfs,myL2), fontsize=10, y=0.95)

    
    def update_CFS_data_anim(idx):
        k=df.image_idx.iloc[idx]
        myL2=df.L2.iloc[idx]
        # Update the data for the plot
        
        occ2d=get_2D_occ_map_from_seq('%s/ising2D_seq_%s.dat'%(seq_dir,str(k).zfill(6)))
        axes[0].imshow(np.transpose(occ2d),interpolation='nearest',cmap='gray') 
        axes[0].axis("off")
        imdat = read_bin('%s/hk0_%s.bin'%(bin_dir,str(k).zfill(6)), npixels=64, offset=1280)
        axes[1].imshow(np.flip(imdat,0),cmap='gray')
        axes[1].axis("off")
        corrf=df.iloc[idx][1:(cfs*cfs+1)].values
      
        sns.heatmap(corrf.reshape((cfs,-1)), ax=axes[2], cmap='bwr',square=True, annot=False, cbar=0  ) 
  
        axes[2].axis('off')
        fig.suptitle("CFS%d vector L2 distance from origin %.3f " %(cfs,myL2), fontsize=10, y=0.95)
        
    fig, axes = plt.subplots(1, 3, figsize=(8,3))
    ani = FuncAnimation(fig, update_CFS_data_anim, init_func=init_CFS_data_anim(), frames=nframes, interval=200)
    return ani


def check_cfs_dist(df,cfs=4):

    my_cols=df.columns.values
    palette = sns.color_palette("Set2", n_colors=cfs*cfs)
    rows,cols =cfs,cfs 
    fig, axes = plt.subplots(rows, cols, figsize=(15,6))
    k=0
    for i in range(rows): 
        for j in range(cols):
            mycol=my_cols[k]
            sns.histplot(ax=axes[i,j], data=df, x=mycol, bins=50,color=palette[k] , kde=True, stat='density', label='%s'%mycol )
            axes[i,j].set_xlabel(None)
            axes[i,j].set_ylabel(None)
            axes[i,j].set_xlim([-1,1])
            axes[i,j].legend(loc=1)
            k+=1
            if k == len(my_cols): break

    fig.suptitle("Density distributions of the randomly generated target CFS4 variables used for CNN fit.", fontsize=12, y=0.95)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)

    axes[0, 0].remove()  # remove unused subplot
# plt.legend()


def prep_img_data(h5name,csv_name,cfs=4):

     # Open the HDF5 file in read-only mode
    with h5py.File(h5name, 'r') as f:
         # Get a list of dataset names in the HDF5 file
            dataset_names = list(f.keys())
            for name in dataset_names:
                print(name)
            dset=f[dataset_names[0]]
            X=dset[:]
         
         # read in  the correlation datum  
    df=pd.read_csv(csv_name)
    df=df.drop('Unnamed: 0',axis=1)
    df.reset_index(inplace=True,drop=True)
    print(df.head())
    ncols=cfs*cfs 
    y=df.iloc[:,1:ncols].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.2)


    X_train=np.expand_dims(X_train, -1)
    X_test=np.expand_dims(X_test, -1)

    print(np.shape(X_train),type(X_train))
    print(np.shape(X_test),type(X_test))
    print(np.shape(y_train),type(y_train))
    print(np.shape(y_test),type(y_test))

    ax=plt.hist(np.log(X.flatten()),bins=1000,log=False,density=True)
    print(np.mean(X.flatten()))
    print(np.median(X.flatten()))
    print(np.max(X.flatten()))
    print(np.min(X.flatten()))
    
    return df, X, y, X_train, X_test, y_train, y_test


def get_metrics_ypred_ytest(xvar,y_test,y_pred):
    corr_test=y_test[xvar]
    corr_pred=y_pred[xvar]
    r2y = r2_score(corr_test, corr_pred)
    msey=mean_squared_error(corr_test, corr_pred)
    maey=mean_absolute_error(corr_test, corr_pred)
    return [r2y,msey,maey]    


def get_metrics_compare_regens_test_set(xvar,X_test,y_test,exp=1):   
    corr_in=y_test[xvar]
    corr_in=np.r_[1.0,corr_in]
    corr_out=np.loadtxt('./expfiles_%d/corr.out'%exp)
    imdat_0 = X_test[xvar].reshape((64,-1))
    imdat_1 = read_bin('./expfiles_%d/hk0.bin'%exp, npixels=64, offset=1280)
    
  #  print(corr_in, corr_out.flatten())
  #  print(np.shape(imdat_0)) 
  #  print(np.shape(imdat_1))
    r2c = r2_score(corr_in.flatten(), corr_out.flatten())
    msec=mean_squared_error(corr_in,corr_out.flatten())
    maec=mean_absolute_error(corr_in,corr_out.flatten())

    r2i=r2_score(imdat_0.flatten(),imdat_1.flatten())
    msei=mean_squared_error(imdat_0.flatten(),imdat_1.flatten())
    maei=mean_absolute_error(imdat_0.flatten(),imdat_1.flatten())
    return [r2c,msec,maec,r2i,msei,maei]    



# Create a figure and axis for the plot
# Define a function to update the plot
def regenerate_test_cfs_vector_and_compare(calc_diffuse_func,X_test,y_test,xvar,cfs=4,iconc=0.5,icycles=500,ianneal=500):
    
    rows,cols =1,3 
    fig, axes = plt.subplots(rows, cols, figsize=(6,6))

   
    imdat = X_test[xvar].reshape((64,-1))
    axes[0].imshow(np.flip(imdat,0),cmap='gray')
    axes[0].axis("off")

    
    corrin=y_test[xvar]
    
    corrin=np.r_[1.0,corrin]
   # print(corrin)
    corrin=corrin.reshape((cfs,cfs))
   # print(corrin)
    fhout=open('corr.in','w')
    for row in range(cfs):
        fhout.write(' '.join(["%.6f"%i for i in corrin[row]])+'\n')
    fhout.close()

    
    calc_diffuse_func(iconc,1,icycles,ianneal,1) # do the regen on the image  
    
    occ3D=get_occ_map('./expfiles_1/ising2D_occ.txt')
    axes[1].imshow(np.transpose(occ3D[:,:,0]),interpolation='nearest',cmap='gray') 
    axes[1].axis("off")
    imdat = read_bin('./expfiles_1/hk0.bin', npixels=64, offset=1280)
    axes[2].imshow(np.flip(imdat,0),cmap='gray')
    axes[2].axis("off")
    
    my_metrics=get_metrics_compare_regens_test_set(xvar,X_test,y_test)
    fig.suptitle(" metrics(r2, mse, mae)\n corrfunc: %.3f, %.3f, %.3f \n FT_metrics: %.3f, %.3f, %.3f " %(tuple(my_metrics)), fontsize=10,y=0.73)
    
    
def regenerate_pred_cfs_vector_and_compare(calc_diffuse_func,X_test,y_test,y_pred,xvar,cfs=4,iconc=0.5,icycles=500,ianneal=500):
    rows,cols =1,3 
    fig, axes = plt.subplots(rows, cols, figsize=(6,6))
   
    imdat = X_test[xvar].reshape((64,-1))
    axes[0].imshow(np.flip(imdat,0),cmap='gray')
    axes[0].axis("off")

    corrin=y_pred[xvar]
    corrin=np.r_[1.0,corrin]
    corrin=corrin.reshape((cfs,cfs))
    fhout=open('corr.in','w')
    for row in range(cfs):
        fhout.write(' '.join(["%.6f"%i for i in corrin[row]])+'\n')
    fhout.close()
    
    
    calc_diffuse_func(iconc,1,icycles,ianneal,1) # do the regen on the image  
    
    occ3D=get_occ_map('./expfiles_1/ising2D_occ.txt')
    axes[1].imshow(np.transpose(occ3D[:,:,0]),interpolation='nearest',cmap='gray') 
    axes[1].axis("off")
    imdat = read_bin('./expfiles_1/hk0.bin', npixels=64, offset=1280)
    axes[2].imshow(np.flip(imdat,0),cmap='gray')
    axes[2].axis("off")
    
    my_metrics=get_metrics_compare_regens_test_set(xvar,X_test,y_test)
    my_metrics2=get_metrics_ypred_ytest(xvar,y_test,y_pred)
    fig.suptitle("metrics(r2, mse, mae)\n corrfunc: %.3f, %.3f, %.3f \n FT_metrics: %.3f, %.3f, %.3f" %tuple(my_metrics) + "\n corrfunc test vs. pred: %.3f, %.3f, %.3f" %(my_metrics2[0], my_metrics2[1], my_metrics2[2]), fontsize=10, y=0.73)

def get_metrics_compare_predict_with_regen(test_img,corr_in,exp=1):   

    corr_out=np.loadtxt('./expfiles_%d/corr.out'%exp)
    imdat_0 = test_img.copy()
    imdat_1 = read_bin('./expfiles_%d/hk0.bin'%exp, npixels=64, offset=1280)
    

    r2c = r2_score(corr_in.flatten(), corr_out.flatten())
    msec=mean_squared_error(corr_in.flatten(),corr_out.flatten())
    maec=mean_absolute_error(corr_in.flatten(),corr_out.flatten())

    r2i=r2_score(imdat_0.flatten(),imdat_1.flatten())
    msei=mean_squared_error(imdat_0.flatten(),imdat_1.flatten())
    maei=mean_absolute_error(imdat_0.flatten(),imdat_1.flatten())
    return [r2c,msec,maec,r2i,msei,maei]    

    
    
    
def predict_and_regen_plot(calc_diffuse_func,test_img,corrin,cfs=4,iconc=0.5,icycles=300,ianneal=300):
       
    rows,cols =1,4 
    fig, axes = plt.subplots(rows, cols, figsize=(14,6))

    axes[0].imshow(np.flip(test_img.reshape((64,-1)),0),cmap='gray')
    axes[0].axis("off")
    
    corrin=np.r_[1.0,corrin]
    corrin=corrin.reshape((cfs,cfs))
    fhout=open('corr.in','w')
    for row in range(cfs):
        fhout.write(' '.join(["%.6f"%i for i in corrin[row]])+'\n')
    fhout.close()
    
    
    calc_diffuse_func(iconc,1,icycles,ianneal,1) # do the regen on the image  
    
    sns.heatmap(np.transpose(corrin), ax=axes[3], cmap='bwr', annot=True, square=True, vmin=-1,vmax=1.0,fmt=".3f" ,cbar=0 )
    axes[3].axis('off')
    
    occ3D=get_occ_map('./expfiles_1/ising2D_occ.txt')
    axes[1].imshow(np.transpose(occ3D[:,:,0]),interpolation='nearest',cmap='gray') 
    axes[1].axis("off")
    imdat = read_bin('./expfiles_1/hk0.bin', npixels=64, offset=1280)
    axes[2].imshow(np.flip(imdat,0),cmap='gray')
    # axes[2].imshow(imdat,cmap='gray')
    axes[2].axis("off")
    
    my_metrics=get_metrics_compare_predict_with_regen(test_img.reshape((64,-1)),corrin,exp=1)
    fig.suptitle("metrics(r2, mse, mae)\n corrfunc: %.3f, %.3f, %.3f \n FT_metrics: %.3f, %.3f, %.3f " %(tuple(my_metrics)), fontsize=14,y=0.88)
    
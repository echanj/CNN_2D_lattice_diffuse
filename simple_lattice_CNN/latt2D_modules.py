import numpy as np 
from matplotlib import cm
from matplotlib import pyplot as plot 
from matplotlib.colors import ListedColormap

import os
import re
import subprocess 
import shutil
from shutil import copyfile, copy2
from shutil import move

def output_16bit_pgm(name,image,mode):  # reads in image array and outputs pgm

 pixels_y,pixels_x = np.shape(image)

# pgmheader = 'P2\n'+'%d %d\n'%(pixels_x,pixels_y)+'65535'

 if mode:
 # this part outputs a pgm
  pgmout = open(name,'wb')
  pgmout.write('P5\n')
  pgmout.write('%i %i\n' %(pixels_x,pixels_y))
  pgmout.write('65535\n')   # this is for 16bit only 
#  pgmout.write('255\n')   # this is for 8bit only 
  pgmout.write(image[:,:])
  pgmout.close()
 else:
  pgmheader = 'P2\n'+'%d %d\n'%(pixels_x,pixels_y)+'65535'
  np.savetxt(name, image, fmt="%5d",header=pgmheader,comments='')





def read_bin(bnam,npixels,offset):
 '''
in order to get the offset byte setting start with offset=0 and figure out the total length of input binary array
using len().  The offset is always (len(X_with_zero_offset)-n_pixels)*4.0
so for a 200x200 hk0.bin we have [41000-(200x200)]*4=4000  
and for a 64x64 hk0.bin we have [4416-(64x64)]*4=1280 

if resizing the array remember to delete the old .bin
becasue diffuse will just reread into a pre-existing file.
we want to generate a new one.
 '''


 fhin=open(bnam,'rb')
 data = np.fromfile(fhin, dtype=np.float32, offset=offset)
 # print (np.shape(data))
 imdat = np.reshape(data,(npixels,-1))
 #print (np.shape(imdat))

 # fliped when viewing so the origin is bottom right hand corner.
 # plot.figure()
 # plot.imshow(np.flip(imdat,0),cmap='gray')  
 # plot.show()

 return imdat


# this is an option to spit out a sorted occfile
# which might be used ofr comparision or needed when sorted input is specific 
# for a particular program
def sort_occfile(inocc,asizein,bsizein,csizein,lsizein,zmax,mmax):
 
 occ4D=np.zeros((asizein,bsizein,csizein,lsizein))    
 for i in range(np.shape(inocc)[0]):
     a=int(inocc[i,0])
     b=int(inocc[i,1])
     c=int(inocc[i,2])
     l=int(inocc[i,3])
     occ4D[a-1,b-1,c-1,l-1]=int(inocc[i,4])

 fhout=open('occ_sorted.txt','wb')
 for a in range(asizein): 
  for b in range(bsizein): 
   for c in range(csizein): 
    for l in range(lsizein):
     fhout.write("%8i%8i%8i%8i%8i%8i\n"%(a+1,b+1,c+1,l+1,occ4D[a,b,c,l],1)) 
 fhout.close()

def plot_occ_map(fname):

# from sys import argv
# script,  fname = argv   
 
 inocc = np.loadtxt(fname)

 # get the extents 
 asizein = int(np.max(inocc[:,0]))
 bsizein = int(np.max(inocc[:,1]))
 csizein = int(np.max(inocc[:,2]))
 lsizein = int(np.max(inocc[:,3]))
 zmax  = int(np.max(inocc[:,4]))
 mmax  = int(np.max(inocc[:,5]))

# sort_occfile(inocc,asizein,bsizein,csizein,lsizein,zmax,mmax)

 # map locations to 3d model here
 asize=int(asizein)
 bsize=int(bsizein)
 csize=int(csizein*lsizein)

 # assemble 3d model
 occ3D=np.zeros((asize,bsize,csize))   # this part is manual 
 print ("shape of input occ.txt %i %i %i" %(np.shape(inocc)[0],np.shape(inocc)[1],np.size(inocc))  )
 print ("shape of 3D array  %i %i %i %i" %(np.shape(occ3D)[0],np.shape(occ3D)[1],np.shape(occ3D)[2],np.size(occ3D)))

 entry = 0 
 for a in range(asizein): 
  for b in range(bsizein): 
   for c in range(csizein): 
   # for l in range(lsizein): 
     z = inocc[entry,4]
 #  manuall maping to 3D array also depends on structure here
     apos=a
     bpos=b
     cpos=c        
     occ3D[apos,bpos,cpos]=z
 #    print a ,b , int(c*4+l) 
     entry += 1
 
#  occ3D[entry[0]-1,entry[1]-1,(entry[2]-1)+(entry[3]-1)]=entry[4] 
 
# plot.figure() 
# plot.imshow(occ3D[:,:,0],interpolation='nearest',cmap=cm.Greys) 
# plot.show()

 a = 0.8

 # Get the colormap colors, multiply them with the factor "a", and create new colormap
# my_cmap = plot.cm.BrBG(np.arange(plot.cm.BrBG.N))
# my_cmap = plot.cm.summer(np.arange(plot.cm.summer.N))
 my_cmap = plot.cm.gray(np.arange(plot.cm.gray.N))

 # hackjob by EJC 
 # the offset has the effect of brightening the colormap 
 # so that is roughly looks like what I wanted 
 my_cmap[:,0:3] *= a 
# my_cmap[:,0:3] += 0.2

 my_cmap = ListedColormap(my_cmap)


 plot.figure() 
 plot.imshow(np.transpose(occ3D[:,:,0]),interpolation='nearest',cmap=my_cmap) 
# plot.title('(001) face') #add titles to axis
 plot.xlabel('a-axis')
 plot.ylabel('b-axis')
# plot.axvline(0.5,linestyle='--',color='black')  # the increment here are manually set for the diagnostic
# plot.axvline(1.5,linestyle='--',color='black')
# plot.axhline(0.5,linestyle='--',color='black')  # the increment here are manually set for the diagnostic
# plot.axhline(1.5,linestyle='--',color='black')
# plot.axis('off') # using this only when working with powerpoint figures 

 plot.savefig("occupancy_map.png",dpi=150) 
# plot.show()

# map_limits = [-7,7,-7,7] #the extent of calculated map. in current case it is 5 unit cells
 # cax=ax[1].imshow(model_pdf[:,:,z],interpolation='nearest',cmap=cm.Spectral_r,extent=map_limits) 
 # cax=ax.imshow(model_pdf[:,:,z],interpolation='nearest',cmap=cm.gist_gray_r,extent=map_limits) 
# cax.set_clim((-c,c))
# cbar = fig.colorbar(cax,ax=ax[1]) #setup colorbar
# cbar.set_label('P/Pmax')
# savefig('obs_data.png')
 




def calc_diffuse( iconc = 0.5, cread=1,  icycles = 500, ianneal = 500, exp_no=0):
 
 
 import subprocess 
 import shutil
# experiment arguments 

# simsize = [20,20,20]
# simsize = [50,50]
 
# rootname = 'nsp212121_relabel'                        # transforming to N5  
# contacts_name = 'managed_contacts_fixed_revised.all'   # name of the contacts file
# pgm_mf = './pgm/'                            # pgm master folder - all pgm files from the experiemnt run will be moved here  

# delete the old folder if it exists 
 dir_path = './expfiles_'+str(exp_no)+'/'

 if os.path.exists('./expfiles_'+str(exp_no)):
#        print ( "deleting the previous ZMC expfiles_%i folder"%exp_no)
         shutil.rmtree('./expfiles_'+str(exp_no))  #
 os.makedirs('./expfiles_'+str(exp_no))

# write corr.in and jswitch.in


 # note: for subprocess in python 3 functionality and backwards compatibility
 # add in argument universal_newlines=True  fixes up I/O binary data stream  
 # also added isingrun.stdin.wrtie.flush() which was not needed in python 2.7 
 if cread==1: 
  copy2('corr.in',dir_path)
  copy2('jswitch.in',dir_path)

 iseed1 = np.random.randint(1,31327)
 iseed2 = np.random.randint(1,30080)

 errlog = open(dir_path+'err.log', 'wb')

 ilogfile = open(dir_path+'ising.log', 'wb')
 
 isingrun = subprocess.Popen(['../../ising_2D/ising_2D' ],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=ilogfile,stderr=errlog)
 isingrun.stdin.write('%s\n'%(str(iconc)))
 isingrun.stdin.write('%d\n'%(cread))
 isingrun.stdin.write('corr.in\n') 
 isingrun.stdin.write('jswitch.in\n') 
 isingrun.stdin.write('%s\n' %(str(icycles)))
 isingrun.stdin.write('%s\n' %(str(ianneal)))
 isingrun.stdin.write('%s\n' %(str(iseed1)))
 isingrun.stdin.write('%s\n' %(str(iseed2)))
 isingrun.stdin.flush()     
 isingrun.wait()
 ilogfile.close()
 

 copy2('latt2d_ZMC.inp',dir_path)
 copy2('contacts.all',dir_path)
 copy2('carbon.zmat',dir_path)
 copy2('atom1.qxyz',dir_path)
 copy2('oxygen.zmat',dir_path)
 copy2('atom2.qxyz',dir_path)
 copy2('diffuse_hk0.in',dir_path)

 zmclogfile = open(dir_path+'zmc.log', 'wb')
 zmcrun = subprocess.Popen(['ZMC','--crystal','--diffuse', 'latt2d_ZMC.inp' ],cwd=dir_path,stdout=zmclogfile,stderr=errlog)
 zmcrun.wait()
 zmclogfile.close()

 dzmclogfile = open(dir_path+'diffuse.log', 'wb')
 dzmcrun = subprocess.Popen(['DZMC','latt2d_ZMC.diffuse'],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=dzmclogfile,stderr=errlog)
 dzmcrun.stdin.write('%s\n' %('diffuse_hk0.in'))
 dzmcrun.stdin.write('%s\n' %('hk0.bin'))
 dzmcrun.stdin.flush()     
 dzmcrun.wait()

 dzmclogfile.close()

 errlog.close()

def calc_diffuse_cfs3( iconc = 0.5, cread=1,  icycles = 500, ianneal = 500, exp_no=0):
 
 
 import subprocess 
 import shutil
# experiment arguments 

# simsize = [20,20,20]
# simsize = [50,50]
 
# rootname = 'nsp212121_relabel'                        # transforming to N5  
# contacts_name = 'managed_contacts_fixed_revised.all'   # name of the contacts file
# pgm_mf = './pgm/'                            # pgm master folder - all pgm files from the experiemnt run will be moved here  

# delete the old folder if it exists 
 dir_path = './expfiles_'+str(exp_no)+'/'

 if os.path.exists('./expfiles_'+str(exp_no)):
#        print ( "deleting the previous ZMC expfiles_%i folder"%exp_no)
         shutil.rmtree('./expfiles_'+str(exp_no))  #
 os.makedirs('./expfiles_'+str(exp_no))

# write corr.in and jswitch.in


 # note: for subprocess in python 3 functionality and backwards compatibility
 # add in argument universal_newlines=True  fixes up I/O binary data stream  
 # also added isingrun.stdin.wrtie.flush() which was not needed in python 2.7 
 if cread==1: 
  copy2('corr.in',dir_path)
  copy2('jswitch.in',dir_path)

 iseed1 = np.random.randint(1,31327)
 iseed2 = np.random.randint(1,30080)

 errlog = open(dir_path+'err.log', 'wb')

 ilogfile = open(dir_path+'ising.log', 'wb')
 
 isingrun = subprocess.Popen(['../../ising_2D/ising_2D_cfs3' ],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=ilogfile,stderr=errlog)
 isingrun.stdin.write('%s\n'%(str(iconc)))
 isingrun.stdin.write('%d\n'%(cread))
 isingrun.stdin.write('corr.in\n') 
 isingrun.stdin.write('jswitch.in\n') 
 isingrun.stdin.write('%s\n' %(str(icycles)))
 isingrun.stdin.write('%s\n' %(str(ianneal)))
 isingrun.stdin.write('%s\n' %(str(iseed1)))
 isingrun.stdin.write('%s\n' %(str(iseed2)))
 isingrun.stdin.flush()     
 isingrun.wait()
 ilogfile.close()
 

 copy2('latt2d_ZMC.inp',dir_path)
 copy2('contacts.all',dir_path)
 copy2('carbon.zmat',dir_path)
 copy2('atom1.qxyz',dir_path)
 copy2('oxygen.zmat',dir_path)
 copy2('atom2.qxyz',dir_path)
 copy2('diffuse_hk0.in',dir_path)

 zmclogfile = open(dir_path+'zmc.log', 'wb')
 zmcrun = subprocess.Popen(['ZMC','--crystal','--diffuse', 'latt2d_ZMC.inp' ],cwd=dir_path,stdout=zmclogfile,stderr=errlog)
 zmcrun.wait()
 zmclogfile.close()

 dzmclogfile = open(dir_path+'diffuse.log', 'wb')
 dzmcrun = subprocess.Popen(['DZMC','latt2d_ZMC.diffuse'],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=dzmclogfile,stderr=errlog)
 dzmcrun.stdin.write('%s\n' %('diffuse_hk0.in'))
 dzmcrun.stdin.write('%s\n' %('hk0.bin'))
 dzmcrun.stdin.flush()     
 dzmcrun.wait()

 dzmclogfile.close()

 errlog.close()

def calc_diffuse_cfs4( iconc = 0.5, cread=1,  icycles = 500, ianneal = 500, exp_no=0):
 
 
 import subprocess 
 import shutil
# experiment arguments 

# simsize = [20,20,20]
# simsize = [50,50]
 
# rootname = 'nsp212121_relabel'                        # transforming to N5  
# contacts_name = 'managed_contacts_fixed_revised.all'   # name of the contacts file
# pgm_mf = './pgm/'                            # pgm master folder - all pgm files from the experiemnt run will be moved here  

# delete the old folder if it exists 
 dir_path = './expfiles_'+str(exp_no)+'/'

 if os.path.exists('./expfiles_'+str(exp_no)):
#        print ( "deleting the previous ZMC expfiles_%i folder"%exp_no)
         shutil.rmtree('./expfiles_'+str(exp_no))  #
 os.makedirs('./expfiles_'+str(exp_no))

# write corr.in and jswitch.in


 # note: for subprocess in python 3 functionality and backwards compatibility
 # add in argument universal_newlines=True  fixes up I/O binary data stream  
 # also added isingrun.stdin.wrtie.flush() which was not needed in python 2.7 
 if cread==1: 
  copy2('corr.in',dir_path)
  copy2('jswitch.in',dir_path)

 iseed1 = np.random.randint(1,31327)
 iseed2 = np.random.randint(1,30080)

 errlog = open(dir_path+'err.log', 'wb')

 ilogfile = open(dir_path+'ising.log', 'wb')
 
 isingrun = subprocess.Popen(['../../ising_2D/ising_2D_cfs4' ],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=ilogfile,stderr=errlog)
 isingrun.stdin.write('%s\n'%(str(iconc)))
 isingrun.stdin.write('%d\n'%(cread))
 isingrun.stdin.write('corr.in\n') 
 isingrun.stdin.write('jswitch.in\n') 
 isingrun.stdin.write('%s\n' %(str(icycles)))
 isingrun.stdin.write('%s\n' %(str(ianneal)))
 isingrun.stdin.write('%s\n' %(str(iseed1)))
 isingrun.stdin.write('%s\n' %(str(iseed2)))
 isingrun.stdin.flush()     
 isingrun.wait()
 ilogfile.close()
 

 copy2('latt2d_ZMC.inp',dir_path)
 copy2('contacts.all',dir_path)
 copy2('carbon.zmat',dir_path)
 copy2('atom1.qxyz',dir_path)
 copy2('oxygen.zmat',dir_path)
 copy2('atom2.qxyz',dir_path)
 copy2('diffuse_hk0.in',dir_path)

 zmclogfile = open(dir_path+'zmc.log', 'wb')
 zmcrun = subprocess.Popen(['ZMC','--crystal','--diffuse', 'latt2d_ZMC.inp' ],cwd=dir_path,stdout=zmclogfile,stderr=errlog)
 zmcrun.wait()
 zmclogfile.close()

 dzmclogfile = open(dir_path+'diffuse.log', 'wb')
 dzmcrun = subprocess.Popen(['DZMC','latt2d_ZMC.diffuse'],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=dzmclogfile,stderr=errlog)
 dzmcrun.stdin.write('%s\n' %('diffuse_hk0.in'))
 dzmcrun.stdin.write('%s\n' %('hk0.bin'))
 dzmcrun.stdin.flush()     
 dzmcrun.wait()

 dzmclogfile.close()

 errlog.close()

def calc_diffuse_cfs4_big( iconc = 0.5, cread=1,  icycles = 500, ianneal = 500, exp_no=0):
 
 
 import subprocess 
 import shutil
# experiment arguments 

# simsize = [20,20,20]
# simsize = [50,50]
 
# rootname = 'nsp212121_relabel'                        # transforming to N5  
# contacts_name = 'managed_contacts_fixed_revised.all'   # name of the contacts file
# pgm_mf = './pgm/'                            # pgm master folder - all pgm files from the experiemnt run will be moved here  

# delete the old folder if it exists 
 dir_path = './expfiles_'+str(exp_no)+'/'

 if os.path.exists('./expfiles_'+str(exp_no)):
#        print ( "deleting the previous ZMC expfiles_%i folder"%exp_no)
         shutil.rmtree('./expfiles_'+str(exp_no))  #
 os.makedirs('./expfiles_'+str(exp_no))

# write corr.in and jswitch.in


 # note: for subprocess in python 3 functionality and backwards compatibility
 # add in argument universal_newlines=True  fixes up I/O binary data stream  
 # also added isingrun.stdin.wrtie.flush() which was not needed in python 2.7 
 if cread==1: 
  copy2('corr.in',dir_path)
  copy2('jswitch.in',dir_path)

 iseed1 = np.random.randint(1,31327)
 iseed2 = np.random.randint(1,30080)

 errlog = open(dir_path+'err.log', 'wb')

 ilogfile = open(dir_path+'ising.log', 'wb')
 
 isingrun = subprocess.Popen(['../../ising_2D/ising_2D_cfs4_big' ],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=ilogfile,stderr=errlog)
 isingrun.stdin.write('%s\n'%(str(iconc)))
 isingrun.stdin.write('%d\n'%(cread))
 isingrun.stdin.write('corr.in\n') 
 isingrun.stdin.write('jswitch.in\n') 
 isingrun.stdin.write('%s\n' %(str(icycles)))
 isingrun.stdin.write('%s\n' %(str(ianneal)))
 isingrun.stdin.write('%s\n' %(str(iseed1)))
 isingrun.stdin.write('%s\n' %(str(iseed2)))
 isingrun.stdin.flush()     
 isingrun.wait()
 ilogfile.close()
 

 copy2('./zmc_inputs/latt2d_ZMC_96.inp',dir_path)
 copy2('contacts.all',dir_path)
 copy2('carbon.zmat',dir_path)
 copy2('atom1.qxyz',dir_path)
 copy2('oxygen.zmat',dir_path)
 copy2('atom2.qxyz',dir_path)
 copy2('./zmc_inputs/diffuse_hk0_96.in',dir_path)

 zmclogfile = open(dir_path+'zmc.log', 'wb')
 zmcrun = subprocess.Popen(['ZMC','--crystal','--diffuse', 'latt2d_ZMC_96.inp' ],cwd=dir_path,stdout=zmclogfile,stderr=errlog)
 zmcrun.wait()
 zmclogfile.close()

 dzmclogfile = open(dir_path+'diffuse.log', 'wb')
 dzmcrun = subprocess.Popen(['DZMC','latt2d_ZMC_96.diffuse'],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=dzmclogfile,stderr=errlog)
 dzmcrun.stdin.write('%s\n' %('diffuse_hk0_96.in'))
 dzmcrun.stdin.write('%s\n' %('hk0.bin'))
 dzmcrun.stdin.flush()     
 dzmcrun.wait()

 dzmclogfile.close()

 errlog.close()


def recalc_diff_from_other_cfs4_big(exp_source=0,exp_no=1):
 
 
 import subprocess 
 import shutil
# experiment arguments 

# simsize = [20,20,20]
# simsize = [50,50]
 
# rootname = 'nsp212121_relabel'                        # transforming to N5  
# contacts_name = 'managed_contacts_fixed_revised.all'   # name of the contacts file
# pgm_mf = './pgm/'                            # pgm master folder - all pgm files from the experiemnt run will be moved here  

# delete the old folder if it exists 
 dir_path = './expfiles_'+str(exp_no)+'/'

 if os.path.exists('./expfiles_'+str(exp_no)):
#        print ( "deleting the previous ZMC expfiles_%i folder"%exp_no)
         shutil.rmtree('./expfiles_'+str(exp_no))  #
 os.makedirs('./expfiles_'+str(exp_no))

# write corr.in and jswitch.in



 errlog = open(dir_path+'err.log', 'wb')

 

 copy2('./expfiles_'+str(exp_source)+'/ising2D_occ.txt',dir_path)
 copy2('./expfiles_'+str(exp_source)+'/corr.out',dir_path)
 copy2('./expfiles_'+str(exp_source)+'/ising.log',dir_path)
 copy2('./zmc_inputs/latt2d_ZMC_96.inp',dir_path)
 copy2('contacts.all',dir_path)
 copy2('carbon.zmat',dir_path)
 copy2('atom1.qxyz',dir_path)
 copy2('oxygen.zmat',dir_path)
 copy2('atom2.qxyz',dir_path)
 copy2('./zmc_inputs/diffuse_hk0_96.in',dir_path)

 zmclogfile = open(dir_path+'zmc.log', 'wb')
 zmcrun = subprocess.Popen(['ZMC','--crystal','--diffuse', 'latt2d_ZMC_96.inp' ],cwd=dir_path,stdout=zmclogfile,stderr=errlog)
 zmcrun.wait()
 zmclogfile.close()

 dzmclogfile = open(dir_path+'diffuse.log', 'wb')
 dzmcrun = subprocess.Popen(['DZMC','latt2d_ZMC_96.diffuse'],shell=False,cwd=dir_path,stdin=subprocess.PIPE,universal_newlines=True,stdout=dzmclogfile,stderr=errlog)
 dzmcrun.stdin.write('%s\n' %('diffuse_hk0_96.in'))
 dzmcrun.stdin.write('%s\n' %('hk0.bin'))
 dzmcrun.stdin.flush()     
 dzmcrun.wait()

 dzmclogfile.close()

 errlog.close()


def get_occ_map(fname):

# from sys import argv
# script,  fname = argv   
 
 inocc = np.loadtxt(fname)

 # get the extents 
 asizein = int(np.max(inocc[:,0]))
 bsizein = int(np.max(inocc[:,1]))
 csizein = int(np.max(inocc[:,2]))
 lsizein = int(np.max(inocc[:,3]))
 zmax  = int(np.max(inocc[:,4]))
 mmax  = int(np.max(inocc[:,5]))

# sort_occfile(inocc,asizein,bsizein,csizein,lsizein,zmax,mmax)

 # map locations to 3d model here
 asize=int(asizein)
 bsize=int(bsizein)
 csize=int(csizein*lsizein)

 # assemble 3d model
 occ3D=np.zeros((asize,bsize,csize))   # this part is manual 
# print ("shape of input occ.txt %i %i %i" %(np.shape(inocc)[0],np.shape(inocc)[1],np.size(inocc))  )
# print ("shape of 3D array  %i %i %i %i" %(np.shape(occ3D)[0],np.shape(occ3D)[1],np.shape(occ3D)[2],np.size(occ3D)))

 entry = 0 
 for a in range(asizein): 
  for b in range(bsizein): 
   for c in range(csizein): 
   # for l in range(lsizein): 
     z = inocc[entry,4]
 #  manuall maping to 3D array also depends on structure here
     apos=a
     bpos=b
     cpos=c        
     occ3D[apos,bpos,cpos]=z
 #    print a ,b , int(c*4+l) 
     entry += 1
 
#  occ3D[entry[0]-1,entry[1]-1,(entry[2]-1)+(entry[3]-1)]=entry[4] 
 return occ3D


def store_occ_map_as_seq(target_path,dest_path):
# this stores the output simualtion as a string of ones and zeros
 arr=np.loadtxt(target_path,usecols=[4])
 fhout=open(dest_path,'w')
 fhout.write(''.join(list((arr-1).astype(int).astype(str))))
 fhout.close()


def get_2D_occ_map_from_seq(fname,m=50):
 ''' 
   fname : filename
   m : length of one side 
 '''
 with open(fname,'r') as f:
    myseq=f.read()

 occ2D=np.array(list(myseq)).astype(int).reshape((m,-1))
 
 return occ2D



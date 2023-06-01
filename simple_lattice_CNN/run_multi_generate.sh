#!/bin/bash

rm -r cfs4_big_bin_1
rm -r cfs4_big_seq_1
rm cfs4_big_corr_out_1.csv
rm cfs4_big_bin_1.h5

rm -r cfs4_big_bin_2
rm -r cfs4_big_seq_2
rm cfs4_big_corr_out_2.csv
rm cfs4_big_bin_2.h5


# python generate_cfs_data.py 'my bin dir' 'cfs4_big_seq_2' 'cfs4_big_corr_out_2.csv' exp=2 N=10 

python generate_cfs_data.py  'cfs4_big_bin_1' 'cfs4_big_seq_1' 'cfs4_big_corr_out_1.csv'  2  1000  > log1.out 2>&1 & 
python generate_cfs_data.py  'cfs4_big_bin_2' 'cfs4_big_seq_2' 'cfs4_big_corr_out_2.csv'  3  1000  > log2.out 2>&1 & 
python generate_cfs_data.py  'cfs4_big_bin_3' 'cfs4_big_seq_3' 'cfs4_big_corr_out_3.csv'  4  1000  > log3.out 2>&1 & 
python generate_cfs_data.py  'cfs4_big_bin_4' 'cfs4_big_seq_4' 'cfs4_big_corr_out_4.csv'  5  1000  > log4.out 2>&1 & 
python generate_cfs_data.py  'cfs4_big_bin_5' 'cfs4_big_seq_5' 'cfs4_big_corr_out_5.csv'  6  1000  > log5.out 2>&1 & 

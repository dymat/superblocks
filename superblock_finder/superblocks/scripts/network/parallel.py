"""Parallel superblocks
"""
import os
import subprocess
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np

path_ini_file = '_'
path_python = 'H:/00_code/superblocks/scripts/network/superblock.py'

def my_function(simulation_number):  
    print('simulation_number ' + str(simulation_number))
    bash_command = "python {} {} {}".format(
        str(path_python),
        str(simulation_number),
        str(path_ini_file))

    os.system(bash_command)


# Parallel mode
nr_total_processors = mp.cpu_count()
nr_processors_to_use = 9 #23
segmentation_IDs = list(range(0, 10)) #100))

if __name__ == "__main__":
    with Pool(nr_processors_to_use) as pool:
        pool.map(
            my_function,
            segmentation_IDs,
            chunksize=1)

'''
if batch_mode:
    batch_counter = None
    arguments = sys.argv
    if len(arguments) > 1:
        batch_counter = int(arguments[1]) # batch counter
else:
    batch_counter = None
'''
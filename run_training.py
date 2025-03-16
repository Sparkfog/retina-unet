###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import configparser as ConfigParser



#config file to read from
config = ConfigParser.RawConfigParser()
config.read(r'configuration.txt')
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

run_GPU = 'CUDA_VISIBLE_DEVICES=0 ' if sys.platform != 'win32' else ''

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

print(tf.__version__)


#create a folder for the results
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing")
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print("copy the configuration file in the results folder")
if sys.platform=='win32':
    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

if nohup:
    print("\n2. Run the training on GPU in the background (Windows alternative to nohup)")

    if sys.platform == "win32":
        os.system(
            run_GPU + f' start /b python -u ./src/retinaNN_training.py > ./{name_experiment}/{name_experiment}_training.log 2>&1')
    else:
        os.system(
            run_GPU + f' nohup python -u ./src/retinaNN_training.py > ./{name_experiment}/{name_experiment}_training.nohup 2>&1')
else:
    print("\n2. Run the training on GPU (no nohup)")
    os.system(run_GPU + f' python -u ./src/retinaNN_training.py')

#Prediction/testing is run with a different script

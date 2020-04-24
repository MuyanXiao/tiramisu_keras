# tiramisu_keras
This repository is the implementation of [The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326) (FC-DenseNet) using keras. The implementation was extended and modified starting from the one in [0bserver07/One-Hundred-Layers-Tiramisu
](https://github.com/0bserver07/One-Hundred-Layers-Tiramisu). The pipeline was used for the [lake ice project](https://prs.igp.ethz.ch/research/completed_projects/integrated-monitoring-of-ice-in-selected-swiss-lakes.html). 

##### Installation:
- Numpy
- Keras
- H5py
- Opencv
- Tensorflow

##### Directory structure:
    + tiramisu_keras
    + Data
     + Images
     + Labels
    + Model
    + Result

### Data preparation
1. Write the image data files in to an HDF5 file: 
python saveHDF5.py PATH_TO_IMAGE PATH_TO_ANNOTATION PATH_TO_HDF5 NAME_TO_HDF5
(e.g. python saveHDF5.py ../Data/Images/ ../Data/Labels/ ../Data/ demo.hdf5)

2. Divide the image data into training, validation and testing data sets:
python data_loader.py PATH_TO_IMAGE --hdf5_dir PATH_TO_HDF5 --hdf5_file NAME_TO_HDF5
(e.g. python data_loader.py ../Data/Images/ --hdf5_dir ../Data/ --hdf5_file demo.hdf5)

### Training and evluation
python train.py PATH_TO_IMAGE PATH_TO_HDF5 NAME_OF_HDF5_FILE --dim_patch PATCH_SIZE --pre_trained PATH_NAME_PRETRAINED_MODEL

### Testing
python test.py PATH_TO_STORE_PREDICTION NAME_OF_THE_MODEL DATA_FILE_NAME --dim_patch PATCH_SIZE

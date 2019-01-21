# tiramisu_keras
This repository is the implementation of [The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326) (FC-DenseNet) using keras. The implementation was extended and modified starting from the one in [0bserver07/One-Hundred-Layers-Tiramisu
](https://github.com/0bserver07/One-Hundred-Layers-Tiramisu). The pipeline was used for the [lake ice detection project](http://www.prs.igp.ethz.ch/research/current_projects/integrated-monitoring-of-ice-swiss-lakes.html). 

##### Installation:
- Numpy
- Keras
- H5py
- Opencv

##### Directory structure:
    + tiramisu_keras
    + Data
     + Images
     + Labels
    + Model
    + Result

### Data preparation
Write the image data files in to an HDF5 file: 
python saveHDF5.py PATH_TO_IMAGE PATH_TO_ANNOTATION PATH_TO_HDF5 NAME_TO_HDF5

### Training and evluation
python train.py PATH_TO_IMAGE NAME_TO_HDF5

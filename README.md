# tiramisu_keras

This repository is the implementation (keras) of: 

Xiao M., Rothermel M., Tom M., Galliani S., Baltsavias E., Schindler K.: [Lake Ice Monitoring with Webcams](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2/311/2018/isprs-annals-IV-2-311-2018.pdf), ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, IV-​2, pages 311-​317, 2018

This work is part of the [Lake Ice Project (Phase 1)](https://prs.igp.ethz.ch/research/completed_projects/integrated-monitoring-of-ice-in-selected-swiss-lakes.html). Here is the link to [Phase 2](https://prs.igp.ethz.ch/research/current_projects/integrated-lake-ice-monitoring-and-generation-of-sustainable--re.html) of the same project.

The implementation was extended and modified starting from the one in [0bserver07/One-Hundred-Layers-Tiramisu
](https://github.com/0bserver07/One-Hundred-Layers-Tiramisu).

##### Pre-requisites:
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

## Citation
@Article{muyan_lakeice_2018,
AUTHOR = {Xiao, M. and Rothermel, M. and Tom, M. and Galliani, S. and Baltsavias, E. and Schindler, K.},
TITLE = {Lake ice monitoring with webcams},
JOURNAL = {ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {IV-2},
YEAR = {2018},
PAGES = {311--317},
DOI = {10.5194/isprs-annals-IV-2-311-2018}
}

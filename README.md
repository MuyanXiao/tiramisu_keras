# tiramisu_keras
Implementation of [The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326) (FC-DenseNet) using keras, the pipeline was used for the [lake ice detection project](http://www.prs.igp.ethz.ch/research/current_projects/integrated-monitoring-of-ice-swiss-lakes.html).

### Data preparation
Write the image data files in to an HDF5 file: 
python saveHDF5.py PATH_TO_IMAGE PATH_TO_ANNOTATION PATH_TO_HDF5 NAME_TO_HDF5

### Training and evluation
python train.py PATH_TO_IMAGE NAME_TO_HDF5

### Testing

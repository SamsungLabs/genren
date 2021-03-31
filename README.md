## Cycle-Consistent Generative Rendering for 2D-3D Modality Translation 

This repo is an implementation of the 2D-3D cyclic generative renderer, which performs *modality translation* between textured 3D meshes and masked 2D RGB images with weak supervision.

**Cycle-Consistent Generative Rendering for 2D-3D Modality Translation**  
Tristan Aumentado-Armstrong, Alex Levinshtein, Stavros Tsogkas, Konstantinos Derpanis, Allan Jepson  
International Conference on 3D Vision (3DV 2020)

[website](https://ttaa9.github.io/genren/) |
[models](https://ln.sync.com/dl/13b71fa40/4eyzp337-ixvghei7-t94i5edm-ndx4p2ww) |
[paper](https://ieeexplore.ieee.org/document/9320324) |
[arxiv](https://arxiv.org/abs/2011.08026)

This work was done at the Samsung AI Centre Toronto, and is licensed under [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) (see [below](#license)).

## Usage

### Installation

Clone with submodules
   
    git clone https://github.com/ttaa9/genren.git --recursive

If you forget to clone with `--recursive`, use

    git submodule update --init --recursive

Install the environment

    conda env create -f env.yml

After activating the `genren` environment, you must install `SoftRas` via `python setup.py install`.

Run `python train.py --help` and the usage arguments should be displayed.

### Training a Model

Prepare two directories with data: 
 - For one, just place a set of `png` images within it.
 - For the other, place a set of oriented point clouds inside. In particular, for each PC, place two pytorch objects: `[filename].PC.pt` as the `NP x 3` array of point set coordinates and `[filename].normals.pt` as the corresponding `NP x 3` set of normal vectors.

Add a new dict entry (say, `new`) to the `OPTIONS` variable in `options.py`. This specifies the options that will be used during training. The `BASE` dictionary holds default values, and inheritance between dictionaries is done via the `COPY_FROM` key. 
See the `test` entry as an example.
The hyper-parameters used for ShapeNet are already present. Note that any arguments can be over-written by passing values for them through the argparse interface. 

Set the `img_data_dir` and `shape_data_dir` keys to be paths to the images and shapes directories above, within `new`.

Create an output directory to hold training intermediates (e.g., `mkdir output`). This includes renders and saved models.   
 
Run `python train.py [option] [output_dir]` to initiate training, which would be `new` and `output`, respectively, in our example. 

### Loading a Model

An example of how to load and run a model is shown in `loadtest.py`.
Trained models on ShapeNet are available [here](https://ln.sync.com/dl/13b71fa40/4eyzp337-ixvghei7-t94i5edm-ndx4p2ww).
Place them in a directory, say `Models`, and obtain images and shapes as in the section above. 

Then, using the cabinet model for instance, running 
```
python loadtest.py Models/model-cabinet-latest.state_dict.pt recon_test_v img_recon --options_choice cabinet --imgs_dir cabinet-images --allow_overwrite True
```
will save example reconstructions and re-renders (vision cycle) into directory `recon_test_v`, while
```
python loadtest.py Models/model-cabinet-latest.state_dict.pt recon_test_g shape_recon --options_choice cabinet  --allow_overwrite True --shapes_dir cabinet-shapes
```
will save example generations (graphics cycle) into `recon_test_g`.

Note: evaluations should be done in batches in `train` mode. I've observed minor artifacts in some cases otherwise. 

### Data

See [Pixel2Mesh](https://arxiv.org/abs/1804.01654) ([github](https://github.com/nywang16/Pixel2Mesh)) and [3D-R2N2](https://arxiv.org/abs/1604.00449) ([github](https://github.com/chrischoy/3D-R2N2)) for rendered [ShapeNet](https://shapenet.org/) data. (Refer to the [genren](https://arxiv.org/abs/2011.08026) paper for usage details.)

## License

[<img src="./.storage/by-nc-sa.png" alt="CC-BY-NC-SA" width="180"/>](https://creativecommons.org/licenses/by-nc-sa/4.0/)

&copy; 2021 by Samsung Electronics Canada Inc. Cycle-Consistent Generative Rendering for 2D-3D Modality Translation.  
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.  
To view a copy of this license, visit  
https://creativecommons.org/licenses/by-nc-sa/4.0/


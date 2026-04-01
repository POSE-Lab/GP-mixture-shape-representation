# Shape Representation using Gaussian Process mixture models

<div align="center">
<span style="color:blue; **font-weight:bold;**">
Panagiotis Sapoutzoglou, George Terzakis, Georgios Floros, Maria Pateraki
</span>

Laboratory of Photogrammetry, School of Rural, Surveying and Geoinformatics Engineering,
National Technical University of Athens, Greece

(psapoutzoglou, gterzakis, gfloros, mpateraki)@mail.ntua.gr
</div>

**Abstract**: Traditional explicit 3D representations, such as point clouds and meshes, demand significant storage to capture fine geometric details
and require complex indexing systems for surface lookups, making functional representations an efficient, compact, and continuous
alternative. In this work, we propose a novel, object-specific functional shape representation that models surface geometry with
Gaussian Process (GP) mixture models. Rather than relying on computationally heavy neural architectures, our method is lightweight,
leveraging GPs to learn continuous directional distance fields from sparsely sampled point clouds. We capture complex topologies
by anchoring local GP priors at strategic reference points, which can be flexibly extracted using any structural decomposition
method (e.g. skeletonization, distance-based clustering). Extensive evaluations on the ShapeNetCore and IndustryShapes datasets
demonstrate that our method can efficiently and accurately represent complex geometries.

## Overview

The core functionality of this repo can be summarized in six steps:

- Installation: Set up the Conda environment and install dependencies using the provided instructions.
- Sample points from a 3D model to serve as the training and test sets. This is done by utilizing the script ```preprocessing.py```.
- Train the GP's to represent the shape of the object. This is done by running the ```train.py``` script.
- Evaluate the trained model over the query test points with ```infer.py```.
- Compute evaluation metrics with ```eval.py```.


## Installation

- Clone the repository and setup the conda environment:
```
git clone https://github.com/POSE-Lab/GP-mixture-shape-representation
cd GP-mixture-shape-representation
conda create --name gprep python=3.10
```
- Install requirments.
```
pip install -r requirments.txt
```

## File organization

You have to place your 3D models under ```./data/models/original/{your_class_name}```. You can change the paths to save the data in ```common.py```.


## Sampling - Preprocessing

We define virtual cameras around the object and ray-casting to acquire points lying only on the outer surface of the object. To sample points for the train and test point-clouds run:

```
python preprocessing.py \
--class-name your_class_name \ 
--num-samples 10000 250000 \ #samples for train and test pcds
--normalize
```

After runing the command, the folders ```./data/models/train/{your_class_name}``` and ```./data/models/test/{your_class_name}``` will be created containing the train and test point clouds for your models.

## Training 

To train COBRA to represent the shape of the objects you simply run:

```
python train.py \
--class-name your_class_name \
--init-lr 0.1
--cluster-overlap 0.2 
--num-steps \ 
--min-num-classes 6 \ 
--max-num-classes 12 \ 
--step
```
A full list of commands can be seen running ```python train.py --help``` as we utilize Tyro cli for the arguments.

After training the trained GP models are saved in ```./data/results/{class_name}/c{num_ref_points}/gps```.

## Infer query points 

To infer the query test point you can run:

```
python infer.py --class_name your_class_name --mesh
```

A full list of the subcomands can be found by running ```python infer.py --help```.

# Evaluation

To evaluate the infered point clouds against the ground truth and compute the metrics you can run:

```
python eval.py --class_name class_name
```
Again a full list of the subcomands can be found by running ```python eval.py --help```. By running this script the best models (i.e. the results for the optimal number of reference points with respect to best metric's error) will be exctracted as .ply files in ```./data/models/est_models/{class_name}```.
The total metrics will be saved under ```./data/results/{class_name}/total_scores.csv``` and they will also be printed in the terminall.
```
        Chairs Statistics          
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric    ┃   Mean    ┃  Median   ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cd        │ 0.000194  │ 0.000191  │
│ emd       │ 0.017624  │ 0.014774  │
│ precision │ 83.789200 │ 83.322800 │
│ recall    │ 91.646533 │ 91.149200 │
│ f1_score  │ 87.529484 │ 88.061813 │
└───────────┴───────────┴───────────┘
```







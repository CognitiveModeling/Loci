# Loci

*Loci is an unsupervised disentangled LOCation and Identity tracking system, which excels on the CATER and related object tracking challenges featuring emergent object permanence and stable entity disentanglement via fully unsupervised learning.* | 

Paper: "Learning What and Where - Unsupervised Disentangling Location and Identity Tracking" | [arXiv](https://arxiv.org/abs/2205.13349)


https://user-images.githubusercontent.com/28415607/191936991-f4f7dccf-75cc-439f-818d-9f25c0482ca4.mp4

## Requirements
A suitable [conda](https://conda.io/) environment named `loci` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate loci
```

## Dataset and trained models

A preprocessed CATER dataset together with the 5 trained networks from the paper can be found [here](https://unitc-my.sharepoint.com/:f:/g/personal/iiimt01_cloud_uni-tuebingen_de/Et0PVeCi7OhMuaz60a5RtcMBgS4Sq-fLAZkjNJsDVFgyOw?e=fLh7xN)

The dataset folder (CATER) needs to be copied to ```data/data/video/```

## Interactive GUI


https://user-images.githubusercontent.com/28415607/191945713-fb43df65-0247-459e-9944-f8c3c7331c93.mp4


We provide an interactive GUI to explore the learned representations of the model. The GUI can load the extracted latent state for one slot. In the top left grid the bits of the gestalt code can be flipped, while in the top right image the position can be changed (by clicking or scrolling). The Bottom half of the GUI shows the composition of the background with the reconstructed slot content as well as the entity's RGB repressentation and mask.

Run the GUI (extracted latent states can be found [here](https://unitc-my.sharepoint.com/:f:/g/personal/iiimt01_cloud_uni-tuebingen_de/Et0PVeCi7OhMuaz60a5RtcMBgS4Sq-fLAZkjNJsDVFgyOw?e=fLh7xN)):

```
python -m model.scripts.playground -cfg model/cater.json -background data/data/video/CATER/background.jpg -device 0 -load net2.pt -latent latent-0000-04.pickle
```

## Training

Training can be started with:

```
python -m model.main -train -cfg model/cater.json
```

## Evaluation

A trained model can be evaluated with:

```
python -m model.main -eval -testset -cfg model/cater.json -load net1.pt
```

Images and latent states can be generated using:


```
python -m model.main -save -testset -cfg model/cater.json -load net1.pt
```

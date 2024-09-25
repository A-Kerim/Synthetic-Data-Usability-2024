# Multi-Armed Bandit Approach for Optimizing Training on Synthetic Data


<img src='https://github.com/A-Kerim/Synthetic-Data-Usability-2024/blob/0584347267c14c7686efb7c7ca2d1dcfa964581b/our-metric.png'>
Each synthetic image, Ii, is assigned a usability score U, which is derived from pixel-level and high-level information.

## Abstract
Supervised machine learning methods require large-scale training datasets to perform well in practice. Synthetic data has been showing great progress recently and has been used as a complement to real data. However,
there is still a great urge to assess the usability of synthetically generated data. To this end, we propose a novel _UCB_-based training procedure combined with a dynamic usability metric. Our proposed metric integrates
low-level and high-level information from synthetic images and their corresponding real and synthetic datasets, surpassing existing traditional metrics. Utilizing a _UCB_-based dynamic approach ensures continual enhancement
of model learning. Unlike other approaches, our method effectively adapts to changes in the machine learning model's state and considers the evolving utility of training samples during the training process. We show that our metric
is a simple yet effective way to rank synthetic images based on their usability. Furthermore, we propose a new pipeline for generating synthetic data by integrating a Large Language Model with Stable Diffusion. Our quantitative
results show that we are able to boost the performance of a wide range of supervised classifiers by deploying our approach. Notably, we observed an improvement of up to 10% in classification accuracy compared to traditional
metrics, demonstrating the effectiveness of our approach.

## Code, Datasets, and Simulator
* Our video stabilization code can be downloaded using this link: [Click](https://livelancsac-my.sharepoint.com/:u:/g/personal/kerim_lancaster_ac_uk/EZv0pwUMO0tMnrhSdei63E0BfxG3rApSkRxL29KGkkXxRA?e=7qCYqi)
* The VSNC35Synth and VSAC65Synth datasets can be downloaded using these links: [Part1](https://drive.google.com/file/d/15IX81jZuYrswblobBxx9EF378BESngWb/view?usp=sharing) and [Part2](https://drive.google.com/file/d/1FMpiJUABFNzMv_C3ef9pK31V-__OPlcx/view?usp=sharing)
* The Silver simulator can be downloaded using this link: [Click](https://livelancsac-my.sharepoint.com/:u:/g/personal/kerim_lancaster_ac_uk/EXmc7xfRKzVJj5Ix8P2AmZQB5KE4QcomfryRqGdlSfQxuw)

## Contact Authors
* Abdulrahman Kerim, PostDoc at Surrey University, a.kerim@surrey.ac.uk
* Leandro Soriano Marcolino, Lecturer at Lancaster University, l.marcolino@lancaster.ac.uk

## Licence
* The dataset and the framework are made freely available to academic and non-commercial purposes. They are provided “AS IS” without any warranty.   
* If you use the dataset or the framework feel free to cite our work (paper link will be shared in the future).

## Acknowledgements
A. Kerim was supported by the Faculty of Science and Technology - Lancaster University.

# Online Personalistaion of Deep Learning Classifiers
> This repository is a subset of https://github.com/mberghofer/online-personalisation-of-dl-classifiers

This repository hosts the code necessary to replicate the experiments of [TODO: add publication here].

The experimental design follows a two-phase approach. In the first part ("offline training") a deep learning classifier 
is trained in a leave-one-subject-out manner. This classifier is then seperated in a deep feature extractor and a 
classifier. For the second phase of the experiment, the deep feature extractor is frozen and utilized to extract the 
features for the shallow online-learning capable classifier, i.e. Support Vector Classifier (SVC), 
Passive-Aggressive-Classifier (PAC) or a Random Forrest (RF).

For the first phase, the following files are relevant:
- `get_model.py`
- `load_data.py`
- `run_experiment.py`

For the second phase the following files are used:
- `extract_features.py`
- `online_learning_experiment.py`
- `trigger_online_experiments.sh`

The code in this repository is divided into the two datasets used for experiments:

**hand-activity**

The fine-grained hand activity dataset was recorded by Laput and Harrison and made publicly available on 
their github page: https://github.com/FIGLAB/hand-activities

See also: 

>Laput, G. and Harrison, C., 2019, May. Sensing fine-grained hand activity with smartwatches. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (pp. 1-13).


**pamap2**

The PAMAP2 dataset was introduced in 2012 by Reiss and Stricker in an
effort to address the lack of a commonly used, standard dataset for human activity
recognition. The dataset is online available. 

> Reiss and D. Stricker. Creating and benchmarking a new dataset for phys-
ical activity monitoring. In Proceedings of the 5th International Conference on
PErvasive Technologies Related to Assistive Environments, pages 1–8, 2012.

> Reiss and D. Stricker. Introducing a new benchmarked dataset for activity
monitoring. In 2012 16th International Symposium on Wearable Computers,
pages 108–109. IEEE, 2012.

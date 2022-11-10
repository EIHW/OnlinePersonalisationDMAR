# Online Personalistaion of Deep Learning Classifiers
> This repository is a subset of https://github.com/mberghofer/online-personalisation-of-dl-classifiers

This repository hosts the code necessary to replicate the experiments of the paper titled. If you find the code useful or if you use it your research, please cite:
[TODO: add publication here]

The experimental design follows a two-phase approach. In the first part ("offline training") a deep learning classifier 
is trained in a leave-one-subject-out manner. This classifier is then seperated in a deep feature extractor and a 
classifier. For the second phase of the experiment, the deep feature extractor is frozen and utilized to extract the 
features for the shallow online-learning capable classifier, i.e. Support Vector Classifier (SVC), 
Passive-Aggressive-Classifier (PAC) or a Random Forrest (RF).

Reproduction:

**fine-grained hand activity**

- Download the fine-grained hand activity dataset and the pre-trained model (recorded by Laput and Harrison) https://github.com/FIGLAB/hand-activities
- Unzip all data files in `data/hand-activity/`
- Copy the pre-trained model to `models/`
- Run `python hand-activity/data_exploration/split_dataset.py` to prepare the data for the target split. Data will be stored in `data/hand-activity/restructured_data/`
- Run `python hand-activity/model/run_experiment.py` to train the feature extractor in a loso manner based on rounds 1 and 2 of the dataset. User-specific models and logs will be saved in the `models/` and runs/` directory.
- Run `hand-activity/model/extract_features.py` to extract features with the deep feature extractor. Data will be stored in `data/hand-activity/extracted_features/` 
- Run `hand-activity/model/online_learning_experiment.py` to extract features with the deep feature extractor to run the experiments for online personalisation models

See also: 

>Laput, G. and Harrison, C., 2019, May. Sensing fine-grained hand activity with smartwatches. In Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (pp. 1-13).


**pamap2**

- Download the pamap2 dataset (recorded by Reiss and Stricker) https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
- Unzip all data files in `data/pamap2/`
- Run `pamap2/model/run_experiment.py` to train the feature extractor. Models and logs will be saved in the `models/` and runs/` directory.
- Run `pamap2/model/extract_features.py` to extract features with the deep feature extractor. Data will be stored in `data/pamap2/extracted_features/` 
- Run `pamap2/model/online_learning_experiment.py` to extract features with the deep feature extractor to run the experiments for online personalisation models. Result will be saved in `pamap2/results/`



See also:

> Reiss and D. Stricker. Creating and benchmarking a new dataset for phys-
ical activity monitoring. In Proceedings of the 5th International Conference on
PErvasive Technologies Related to Assistive Environments, pages 1–8, 2012.

> Reiss and D. Stricker. Introducing a new benchmarked dataset for activity
monitoring. In 2012 16th International Symposium on Wearable Computers,
pages 108–109. IEEE, 2012.

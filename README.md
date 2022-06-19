# Anomaly Detection in Time-Series!
*Time-Series Anomaly Detection Comprehensive Benchmark*

This repository updates the comprehensive list of classic and state-of-the-art methods and datasets for Anomaly Decetion in Time-Series. This is part of an onging research at Time Series Analytics Lab, Monash University.


## Datasets

|Name|Number of time series|Dimensions|Anomalies(%)|Full Name     |Description               |Link|
|----|---------------------|----------|------------|--------------|--------------------------|----|
|SWaT|1                    |51        |12.14       |Secure Water Treatment|Collect the normal sensor and actuator data of the plants as the training set, while several attacks are launched to the system in the testing set (including normal and anomalous data). Training contains only normal data. 7 days collected under normal operations and 4 days collected with attack scenarios.|  [Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)  |
|WADI|1                    |123       |5.85        |Water Distribution|Collect the normal sensor and actuator data of the plants as the training set, while several attacks are launched to the system in the testing set (including normal and anomalous data).Training contains only normal data. 14 days were collected under normal operation and 2 days with attack scenarios.|  [Main article](https://dl.acm.org/doi/10.1145/3055366.3055375)  |
|SMD |28                   |38        |4.16        |Server Machine Dataset| Some machines in SMD experienced service change during the data collection period, which leads to severe concept drift in training and testing data. Provides anomaly detection and interpretation labels on the test set for evaluation.|    |
|SMAP|55                   |25        |13.13       |Soil Moisture Active Passive|Labelled dataset. measures how much water is in the top layer of soil, using this information to produce global maps of soil moisture.|    |
|MSL |27                   |55        |10.72       |Satellite and Mars Science Laboratory|Labelled dataset.         |    |
|Orange (Proprietary Data)|1                    |33        |33.72       |              |The collected data come from technical and business indicators from Orange’s advertising network in its website including 27 technical and 6 business measurements.|    |
|ASD |12                   |19        |4.61        |Application Server Dataset|Published in Interfusion. 12 servers with stable services. Metrics characterizing the status of the servers.  Anomalies and their most anomalous dimensions in the ASD testing set have been labelled by system operators based on incident reports and domain knowledge. Moreover, anomalies are cliassified in 3 types|    |
|NAB |                     |          |            |Numenta Anomaly Benchmark|                          |    |


## Models
|Models|Approaches       |Datasets |Year|Sources       |Pulications               |
|------|-----------------|---------|----|--------------|--------------------------|
|GAN   |Graph + Attention|NAB + MAT|2017|[link 1](https://google.com), [link 2](https://www.google.com)|[Folani2017], [Folani2018], [arXiv]|
|GAN   |Graph + Attention|NAB + MAT|2017|link 1, link 2|[Folani2017], [Folani2018]|
|GAN   |Graph + Attention|NAB + MAT|2017|link 1, link 2|[Folani2017], [Folani2018]|
|GAN   |Graph + Attention|NAB + MAT|2017|link 1, link 2|[Folani2017], [Folani2018]|


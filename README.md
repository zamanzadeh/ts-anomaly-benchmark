
# Deep Learning for Time Series Anomaly Detection (Models and Datasets)
*Time-Series Anomaly Detection Comprehensive Benchmark*

This repository updates the comprehensive list of classic and state-of-the-art methods and datasets for Anomaly Detection in Time Series by
- [Zahra Z. Darban](https://scholar.google.com/citations?user=wl6XsiwAAAAJ&hl=en) [Monash University], 
- [Geoffrey I. Webb](https://scholar.google.com/citations?user=_1tpf8AAAAAJ&hl=en) [Monash University], 
- [Shirui Pan](https://scholar.google.com/citations?user=frWRJN4AAAAJ&hl=en) [Griffith University], 
- [Charu C. Aggarwal](https://scholar.google.com/citations?user=x_wsduUAAAAJ&hl=en) [IBM], and 
- [Mahsa Salehi](https://scholar.google.com/citations?user=tTxJ1owAAAAJ&hl=en) [Monash University]

![enter image description here](monash-logo-mono.svg)

If you use this repository in your works, please cite the main article:

[-] Zamanzadeh Darban, Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Salehi, M. (2022). Deep Learning for Time Series Anomaly Detection: A Survey. arXiv e-prints. doi:10.48550/ARXIV.2211.05244 [[arXiv](https://arxiv.org/abs/2211.05244)]

	@ARTICLE{2022arXiv221105244Z,
		author = {Zamanzadeh Darban, Zahra and Webb, Geoffrey I. and Pan, Shirui and Aggarwal, Charu C. and Salehi, Mahsa},
		title = {Deep Learning for Time Series Anomaly Detection: A Survey},
		journal = {arXiv e-prints},
		year = 2022,
		month = Nov,
		eid = {arXiv:2211.05244},
		eprint = {2211.05244},
		url = {https://arxiv.org/abs/2211.05244},
		doi = {10.48550/ARXIV.2211.05244},
	}


## Related Review Papers
1. [Revisiting Time Series Outlier Detection: Definitions and Benchmarks](https://openreview.net/forum?id=r8IvOsnHchr), NeurIPS 2021.
2. [Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress](https://ieeexplore.ieee.org/abstract/document/9537291), TKDE, 2021.
3. [Towards a Rigorous Evaluation of Time-Series Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/20680), AAAI 2022.
4. [Anomaly detection in time series: a comprehensive evaluation](https://dl.acm.org/doi/abs/10.14778/3538598.3538602), VLDB 2022. 


## Datasets/Benchmarks for time series anomaly detection

|Dataset/Benchmark          |Real/Synth|MTS/UTS|# Samples       |# Entities|# Dim|Domain                                    |
|---------------------------|----------|-------|----------------|----------|-----|------------------------------------------|
|[CalIt2](https://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts)|Real      |MTS    |10,080          |2         |2    |Urban events management                   |
|[CAP](https://physionet.org/content/capslpdb/1.0.0/)|Real      |MTS    |921,700,000     |108       |21   |Medical and health                        |
|[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)|Real      |MTS    |2,830,540       |15        |83   |Server machines monitoring                |
|[Credit Card fraud detection](https://www.openml.org/search?type=data&sort=runs&id=1597&status=active)|Real      |MTS    |284,807         |1         |31   |Fraud detectcion                          |
|[DMDS](https://iair.mchtr.pw.edu.pl/Damadics)|Real      |MTS    |725,402         |1         |32   |Industrial Control Systems                |
|[Engine Dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)|Real      |MTS    |NA              |NA        |12   |Industrial control systems                |
|[Exathlon](https://github.com/exathlonbenchmark/exathlon)|Real      |MTS    |47,530          |39        |45   |Server machines monitoring                |
|[GECCO IoT](https://zenodo.org/record/3884398#.Y1NlUtJByRQ)|Real      |MTS    |139,566         |1         |9    |Internet of things (IoT)                  |
|[Genesis](https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning)|Real      |MTS    |16,220          |1         |18   |Industrial control systems                |
|[GHL](https://kas.pr/ics-research/dataset_ghl_1)|Synth     |MTS    |200,001         |48        |22   |Industrial control systems                |
|[IOnsphere](https://search.r-project.org/CRAN/refmans/fdm2id/html/ionosphere.html)|Real      |MTS    |351             |          |32   |Astronomical studies                      |
|[KDDCUP99](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)|Real      |MTS    |4,898,427       |5         |41   |Computer networks                         |
|[Kitsune](https://archive.ics.uci.edu/ml/datasets/Kitsune+Network+Attack+Dataset)|Real      |MTS    |3,018,973       |9         |115  |Computer networks                         |
|[MBD](https://github.com/QAZASDEDC/TopoMAD)|Real      |MTS    |8,640           |5         |26   |Server machines monitoring                |
|[Metro](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)|Real      |MTS    |48,204          |1         |5    |Urban events management                   |
|[MIT-BIH Arrhythmia (ECG)](https://physionet.org/content/mitdb/1.0.0/)|Real      |MTS    |28,600,000      |48        |2    |Medical and health                        |
|[MIT-BIH-SVDB](https://doi.org/10.13026/C2V30W)|Real      |MTS    |17,971,200      |78        |2    |Medical and health                        |
|[MMS](https://github.com/QAZASDEDC/TopoMAD)|Real      |MTS    |4,370           |50        |7    |Server machines monitoring                |
|[MSL](https://github.com/khundman/telemanom)|Real      |MTS    |132,046         |27        |55   |Aerospace                                 |
|[NAB-realAdExchange](https://github.com/numenta/NAB)|Real      |MTS    |9,616           |3         |2    |Business                                  |
|[NAB-realAWSCloudwatch](https://github.com/numenta/NAB)|Real      |MTS    |67,644          |1         |17   |Server machines monitoring                |
|[NASA Shuttle Valve Data](https://cs.fit.edu/~pkc/nasa/data/)|Real      |MTS    |49,097          |1         |9    |Aerospace                                 |
|[OPPORTUNITY](https://archive.ics.uci.edu/ml/datasets/URL+Reputation)|Real      |MTS    |869,376         |24        |133  |Computer networks                         |
|[Pooled Server Metrics (PSM)](https://github.com/eBay/RANSynCoders)|Real      |MTS    |132,480         |1         |24   |Server machines monitoring                |
|[PUMP](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)|Real      |MTS    |220,302         |1         |44   |Industrial control systems                |
|[SMAP](https://github.com/khundman/telemanom)|Real      |MTS    |562,800         |55        |25   |Environmental management                  |
|[SMD](https://github.com/NetManAIOps/OmniAnomaly/)|Real      |MTS    |1,416,825       |28        |38   |Server machines monitoring                |
|[SWAN-SF](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM)|Real      |MTS    |355,330         |5         |51   |Astronomical studies                      |
|[SWaT](http://itrust.sutd.edu.sg/research/testbeds/secure-water-treatment-swat/)|Real      |MTS    |946,719         |1         |51   |Industrial control systems                |
|[WADI](https://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/)|Real      |MTS    |957,372         |1         |127  |Industrial control systems                |
|[NYC Bike](https://ride.citibikenyc.com/system-data)|Real      |MTS/UTS|+25M             |NA        |NA   |Urban events management                   |
|[NYC Taxi](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)|Real      |MTS/UTS|+200M            |NA        |NA   |Urban events management                   |
|[UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018)|Real/Synth|MTS/UTS|NA              |NA        |NA   |Multiple domains                          |
|[Dodgers Loop Sensor Dataset](https://archive.ics.uci.edu/ml/datasets/dodgers+loop+sensor)|Real      |UTS    |50,400          |1         |1    |Urban events management                   |
|[IOPS](https://github.com/iopsai/iops)|Real      |UTS    |2,918,821       |29        |1    |Business                                  |
|[KPI AIOPS](https://competition.aiops-challenge.com/home/competition)|Real      |UTS    |5,922,913       |58        |1    |Business                                  |
|[MGAB](https://github.com/MarkusThill/MGAB/.)|Synth     |UTS    |100,000         |10        |1    |Medical and health                        |
|[MIT-BIH-LTDB](https://doi.org/10.13026/C2KS3F)|Real      |UTS    |67,944,954      |7         |1    |Medical and health                        |
|[NAB-artificialNoAnomaly](https://github.com/numenta/NAB)|Synth     |UTS    |20,165          |5         |1    |-                                         |
|[NAB-artificialWithAnomaly](https://github.com/numenta/NAB)|Synth     |UTS    |24,192          |6         |1    |-                                         |
|[NAB-realKnownCause](https://github.com/numenta/NAB)|Real      |UTS    |69,568          |7         |1    |Multiple domains                          |
|[NAB-realTraffic](https://github.com/numenta/NAB)|Real      |UTS    |15,662          |7         |1    |Urban events management                   |
|[NAB-realTweets](https://github.com/numenta/NAB)|Real      |UTS    |158,511         |10        |1    |Business                                  |
|[NeurIPS-TS](https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic)|Synth     |UTS    |NA              |1         |1    |-                                         |
|[NormA](https://helios2.mi.parisdescartes.fr/~themisp/norma/)|Real/Synth|UTS    |1,756,524       |21        |1    |Multiple domains                          |
|[Power Demand Dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)|Real      |UTS    |35,040          |1         |1    |Industrial control systems                |
|[SensoreScope](https://doi.org/10.5281/zenodo.2654726)|Real      |UTS    |621,874         |23        |1    |Internet of things (IoT)                  |
|[Space Shuttle Dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)|Real      |UTS    |15,000          |15        |1    |Aerospace                                 |
|[Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70&guccounter=1)|Real/Synth|UTS    |572,966         |367       |1    |Multiple domains                          |. 
  
    
 
## Univariate Deep Anomaly Detection Models in Time Series

| A<sup>[1](#Approach)</sup> | MA<sup>[2](#Main)</sup> | Model | Year | Su/Un<sup>[3](#Su)</sup> | Input | P/S<sup>[4](#point)</sup> | Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Forecasting | RNN | LSTM-AD <a href="#ref1" id="ref1">[1]</a> | Year |  Un | P | Point | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/lstm_ad) |
| Forecasting | RNN | DeepLSTM <a href="#ref13" id="ref13">[13]</a> | 2015 | Semi | P | Point | |
| Forecasting | RNN | LSTM RNN <a href="#ref2" id="ref2">[2]</a> | 2015 | Semi | P | Subseq |
| Forecasting | RNN | LSTM-based <a href="#ref3" id="ref3">[3]</a> | 2019 | Un | W | - |
| Forecasting | RNN | TCQSA <a href="#ref4" id="ref4">[4]</a> | 2020 | Su | P | - |
| Forecasting | HTM | Numenta HTM <a href="#ref5" id="ref5">[5]</a> | 2017 | Un | - | - |
| Forecasting | HTM | Multi HTM <a href="#ref6" id="ref6">[6]</a> | 2018 | Un | - | - | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/multi_hmm) |
| Forecasting | CNN | SR-CNN <a href="#ref7" id="ref7">[7]</a> | 2019 | Un | W | Point + Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/sr_cnn) |
| Reconstruction | VAE | Donut <a href="#ref8" id="ref8">[8]</a> | 2018 | Un | W | Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/donut) |
| Reconstruction | VAE | Bagel <a href="#ref10" id="ref10">[10]</a> | 2018 | Un | W | Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/bagel) |
| Reconstruction | VAE | Buzz <a href="#ref9" id="ref9">[9]</a> | 2019 | Un | W | Subseq |
| Reconstruction | AE | EncDec-AD <a href="#ref11" id="ref11">[11]</a> | 2016 | Semi | W | Point | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/encdec_ad) |
  

## Multivariate Deep Anomaly Detection Models in Time Series
	
| A<sup>[1](#Approach)</sup>  | MA<sup>[2](#Main)</sup> | Model | Year | T/S<sup>[3](#temp)</sup> | Su/Un<sup>[4](#Su)</sup>  | Input | Int<sup>[5](#int)</sup> | P/S<sup>[6](#point)</sup> | Code |
|-------|--------|-------|---------|-----------|-------|----------|---------|---------|---------|
| Forecasting | RNN | LSTM-PRED <a href="#ref14" id="ref14">[14]</a> | 2017 | T | Un | W | &#10003; | - | |
| Forecasting | RNN | LSTM-NDT <a href="#ref12" id="ref12">[12]</a> | 2018 | T | Un | W | &#10003; | Subseq | [Github](https://github.com/andrejw27/Multivariate-Time-series-Anomaly-Detection-with-Multi-task-Learning) |
| Forecasting | RNN | LGMAD <a href="#ref15" id="ref15">[15]</a> | 2019 | T | Semi | P | | Point | | 
| Forecasting | RNN | THOC <a href="#ref16" id="ref16">[16]</a> | 2020 | T | Self | W | | Subseq | |
| Forecasting | RNN | AD-LTI <a href="#ref17" id="ref17">[17]</a> | 2020 | T | Un | P | | Point (frame) | |
| Forecasting | CNN | DeepAnt <a href="#ref18" id="ref18">[18]</a> | 2018 | T | Un | W | | Point + Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/deepant) |
| Forecasting | CNN | TCN-ms <a href="#ref19" id="ref19">[19]</a> | 2019 | T | Un | W | | - | |
| Forecasting | CNN | TimesNet <a href="#ref57" id="ref57">[57]</a> | 2023 | T | Semi | W | | Subseq | [Github](https://github.com/thuml/TimesNet) |
| Forecasting | GNN | GDN <a href="#ref20" id="ref20">[20]</a> | 2021 | S | Un | W | &#10003; | - | [Github](https://github.com/d-ailin/GDN) |
| Forecasting | GNN | GTA* <a href="#ref21" id="ref21">[21]</a> | 2021 | ST | Semi | - | | - | [Github](https://github.com/zackchen-lb/GTA?tab=readme-ov-file) |
| Forecasting | GNN | GANF <a href="#ref22" id="ref22">[22]</a> | 2022 | ST | Un | W | | | [Github](https://github.com/EnyanDai/GANF) |
| Forecasting | HTM | RADM <a href="#ref23" id="ref23">[23]</a> | 2018 | T | Un | W | | - | |
| Forecasting | Transformer | SAND <a href="#ref24" id="ref24">[24]</a> | 2018 | T | Semi | W | | - | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/sand) |
| Forecasting | Transformer | GTA* <a href="#ref21" id="ref21">[21]</a> | 2021 | ST | Semi | - | | - | [Github](https://github.com/zackchen-lb/GTA?tab=readme-ov-file) |
| Reconstruction | AE | AE/DAE <a href="#ref25" id="ref25">[25]</a> | 2014 | T | Semi | P | | Point | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/dae) |
| Reconstruction | AE | DAGMM <a href="#ref26" id="ref26">[26]</a> | 2018 | S | Un | P | | Point | [Github](https://github.com/danieltan07/dagmm) |
| Reconstruction | AE | MSCRED <a href="#ref27" id="ref27">[27]</a> | 2019 | ST | Un | W | &#10003; | Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/mscred) |
| Reconstruction | AE | USAD <a href="#ref28" id="ref28">[28]</a> | 2020 | T | Un | W | | Point | [Github](https://github.com/manigalati/usad) |
| Reconstruction | AE | APAE <a href="#ref29" id="ref29">[29]</a> | 2020 | T | Un | W | | - | |
| Reconstruction | AE | RANSynCoders <a href="#ref30" id="ref30">[30]</a> | 2021 | ST | Un | P | &#10003; | Point | [Github](https://github.com/eBay/RANSynCoders) |
| Reconstruction | AE | CAE-Ensemble <a href="#ref31" id="ref31">[31]</a> | 2021 | T | Un | W | | Subseq | [Github](https://github.com/d-gcc/CAE-Ensemble) |
| Reconstruction | AE | AMSL <a href="#ref32" id="ref32">[32]</a> | 2022 | T | Self | W | | - | [Github](https://github.com/zhangyuxin621/AMSL) |
| Reconstruction | AE | ContextDA <a href="#ref58" id="ref58">[58]</a> | 2023 | T | Un | W | | Point + Subseq | |
| Reconstruction | VAE | STORN <a href="#ref35" id="ref35">[35]</a> | 2016 | ST | Un | P | | Point | |
| Reconstruction | VAE | GGM-VAE <a href="#ref36" id="ref36">[36]</a> | 2018 | T | Un | W | | Subseq | |
| Reconstruction | VAE | LSTM-VAE <a href="#ref33" id="ref33">[33]</a> | 2018 | T | Semi | P | | - | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/lstm_vae) |
| Reconstruction | VAE | OmniAnomaly <a href="#ref34" id="ref34">[34]</a> | 2019 | T | Un | W | &#10003; | Point + Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEvalalgorithms/blob/main/omnianomaly) |
| Reconstruction | VAE | VELC <a href="#ref39" id="ref39">[39]</a> | 2019 | T | Un | - | | - | [Github](https://github.com/Vikrant7981/anomaly_detection) |
| Reconstruction | VAE | SISVAE <a href="#ref37" id="ref37">[37]</a> | 2020 | T | Un | W | | Point | |
| Reconstruction | VAE | VAE-GAN <a href="#ref38" id="ref38">[38]</a> | 2020 | T | Semi | W | | Point |  |
| Reconstruction | VAE | TopoMAD <a href="#ref40" id="ref40">[40]</a> | 2020 | ST | Un | W | | Subseq | [Github](https://github.com/QAZASDEDC/TopoMAD) |
| Reconstruction | VAE | PAD <a href="#ref41" id="ref41">[41]</a> | 2021 | T | Un | W | | Subseq | | |
| Reconstruction | VAE | InterFusion <a href="#ref42" id="ref42">[42]</a> | 2021 | ST | Un | W | &#10003; | Subseq | [Github](https://github.com/zhhlee/InterFusion) |
| Reconstruction | VAE | MT-RVAE* <a href="#ref43" id="ref43">[43]</a> | 2022 | ST | Un | W | | - | | |
| Reconstruction | VAE | RDSMM <a href="#ref44" id="ref44">[44]</a> | 2022 | T | Un | W | | Point + Subseq | |
| Reconstruction | GAN | MAD-GAN <a href="#ref45" id="ref45">[45]</a> | 2019 | ST | Un | W | | Subseq | [Github](https://github.com/Guillem96/madgan-pytorch) |
| Reconstruction | GAN | BeatGAN <a href="#ref46" id="ref46">[46]</a> | 2019 | T | Un | W | | Subseq | [Github](https://github.com/hi-bingo/BeatGAN) |
| Reconstruction | GAN | DAEMON <a href="#ref47" id="ref47">[47]</a> | 2021 | T | Un | W | &#10003; | Subseq | |
| Reconstruction | GAN | FGANomaly <a href="#ref48" id="ref48">[48]</a> | 2021 | T | Un | W | | Point + Subseq | |
| Reconstruction | GAN | DCT-GAN* <a href="#ref49" id="ref49">[49]</a> | 2021 | T | Un | W | | - | | |
| Reconstruction | Transformer | Anomaly Transformer <a href="#ref50" id="ref50">[50]</a> | 2021 | T | Un | W | | Subseq | [Github](https://github.com/thuml/Anomaly-Transformer) |
| Reconstruction | Transformer | DCT-GAN* <a href="#ref49" id="ref49">[49]</a> | 2021 | T | Un | W | | - | |
| Reconstruction | Transformer | TranAD <a href="#ref51" id="ref51">[51]</a> | 2022 | T | Un | W | &#10003; | Subseq | [Github]((https://github.com/imperial-qore/TranAD) |
| Reconstruction | Transformer | MT-RVAE* <a href="#ref43" id="ref43">[43]</a> | 2022 | ST | Un | W | | - | |
| Reconstruction | Transformer | Dual-TF <a href="#ref59" id="ref59">[59]</a> | 2024 | T | Un | W | | Point + Subseq | |
| Representation | Transformer | TS2Vec <a href="#ref60" id="ref60">[60]</a> | 2022 | T | Self | P | | Point | [Github](https://github.com/zhihanyue/ts2vec) |
| Representation | CNN | TF-C <a href="#ref61" id="ref61">[61]</a> | 2022 | T | Self | W | | - | [Github](https://github.com/mims-harvard/TFC-pretraining) |
| Representation | CNN | DCdetector <a href="#ref62" id="ref62">[62]</a> | 2023 | ST | Self | W | | Point + Subseq | [Github](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) |
| Representation | CNN | CARLA <a href="#ref63" id="ref63">[63]</a> | 2024 | ST | Self | W | | Point + Subseq | [Github](https://github.com/zamanzadeh/CARLA) |
| Representation | CNN | DACAD <a href="#ref64" id="ref64">[64]</a> | 2024 | ST | Self | W | | Point + Subseq | [Github](https://github.com/zamanzadeh/DACAD) |
| Hybrid | AE | CAE-M <a href="#ref52" id="ref52">[52]</a> | 2021 | ST | Un | W | | Subseq | |
| Hybrid | AE | NSIBF* <a href="#ref53" id="ref53">[53]</a> | 2021 | T | Un | W | | Subseq | [Github](https://github.com/cfeng783/NSIBF) |
| Hybrid | RNN | TAnoGAN <a href="#ref54" id="ref54">[54]</a> | 2020 | T | Un | W | | Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/tanogan) |
| Hybrid | RNN | NSIBF* <a href="#ref53" id="ref53">[53]</a> | 2021 | T | Un | W | | Subseq | [Github](https://github.com/cfeng783/NSIBF) |
| Hybrid | GNN | MTAD-GAT <a href="#ref55" id="ref55">[55]</a> | 2020 | ST | Self | W | &#10003; | Subseq | [Github](https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/mtad_gat) |
| Hybrid | GNN | FuSAGNet <a href="#ref56" id="ref56">[56]</a> | 2022 | ST | Semi | W | | Subseq | [Github](https://github.com/sihohan/FuSAGNet) |

---
<sub><a name="Approach">1</a>: Approach. </sub>

<sub><a name="Main">2</a>: Main Approach. </sub>

<sub><a name="temp">3</a>: Temporal/Spatial

<sub><a name="Su">4</a>: Supervised/Unsupervised | Values: [Su: Supervised, Un: Unsupervised, Semi: Semi-supervised, Self: Self-supervised]. </sub>

<sub><a name="int">5</a>: Interpretability

<sub><a name="point">6</a>: Point/Sub-sequence </sub>
	

## Guidelines to Use Evaluation Metrics for Time Series Anomaly Detection
| **Metrics** | **Value Explanation** | **When to Use** |
| --- | --- | --- |
| **Precision** | Low precision indicates many false alarms (normal instances classified as anomalies). High precision indicates most detected anomalies are actual anomalies, implying few false alarms. | Use when it is crucial to minimize false alarms and ensure that detected anomalies are truly significant. |
| **Recall** | Low recall indicates many true anomalies are missed, leading to undetected critical events. High recall indicates most anomalies are detected, ensuring prompt action on critical events. | Use when it is critical to detect all anomalies, even if it means tolerating some false alarms. |
| **F1** | Low F1 score indicates poor balance between precision and recall, leading to either many missed anomalies and/or many false alarms. High F1 score indicates a good balance, ensuring reliable anomaly detection with minimal misses and false alarms. | Use when a balance between precision and recall is needed to ensure reliable overall performance. |
| **F1<sub>PA</sub> Score** | Low F1<sub>PA</sub> indicates difficulty in accurately identifying the exact points of anomalies. High F1<sub>PA</sub> indicates effective handling of slight deviations, ensuring precise anomaly detection. | Use when anomalies may not be precisely aligned, and slight deviations in detection points are acceptable. |
| **PA%K** | Low PA%K indicates that the model struggles to detect a sufficient portion of the anomalous segment. High PA%K indicates effective detection of segments, ensuring that a significant portion of the segment is identified as anomalous. | Use when evaluating the model's performance in detecting segments of anomalies rather than individual points. |
| **AU-PR** | Low AU-PR indicates poor model performance, especially with imbalanced datasets. High AU-PR indicates strong performance, maintaining high precision and recall across thresholds. | Use when dealing with imbalanced datasets, where anomalies are rare compared to normal instances. |
| **AU-ROC** | Low AU-ROC indicates the model struggles to distinguish between normal and anomalous patterns. High AU-ROC indicates effective differentiation, providing reliable anomaly detection. | Use for a general assessment of the model's ability to distinguish between normal and anomalous instances. |
| **MTTD** | High MTTD indicates significant delays in detecting anomalies. Low MTTD indicates quick detection, allowing prompt responses to critical events. | Use when the speed of anomaly detection is critical, and prompt action is required. |
| **Affiliation** | High value of the affiliation metric indicates a strong overlap or alignment between the detected anomalies and the true anomalies in a time series. | Use when a comprehensive evaluation is required, or the focus is early detection. |
| **VUS** | A lower VUS value indicates better performance, as it means the predicted anomaly signal is closer to the true signal. | Use when a holistic and threshold-free evaluation of TSAD methods is required. |




## References
<a href="#ref1" id="ref1">[1]</a> Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, Puneet Agarwal, et al . 2015. Long short term memory networks for anomaly detection in time
series. In Proceedings of ESANN, Vol. 89. 89–94.  

<a id="ref2" href="#ref2">[2]</a> Loïc Bontemps, Van Loi Cao, James McDermott, and Nhien-An Le-Khac. 2016. Collective anomaly detection based on long short-term memory
recurrent neural networks. In International conference on future data and security engineering. Springer, 141–152.  

<a href="#ref3" id="ref3">[3]</a> Tolga Ergen and Suleyman Serdar Kozat. 2019. Unsupervised anomaly detection with LSTM neural networks. IEEE Transactions on Neural Networks
and Learning Systems 31, 8 (2019), 3127–3141.  

<a href="#ref4" id="ref4">[4]</a> Fan Liu, Xingshe Zhou, Jinli Cao, Zhu Wang, Tianben Wang, Hua Wang, and Yanchun Zhang. 2020. Anomaly detection in quasi-periodic time
series based on automatic data segmentation and attentional LSTM-CNN. IEEE Transactions on Knowledge and Data Engineering (2020).  

<a href="#ref5" id="ref5">[5]</a> Subutai Ahmad, Alexander Lavin, Scott Purdy, and Zuha Agha. 2017. Unsupervised real-time anomaly detection for streaming data. Neurocomputing
262 (2017), 134–147.  

<a href="#ref6" id="ref6">[6]</a> Jia Wu, Weiru Zeng, and Fei Yan. 2018. Hierarchical temporal memory method for time-series-based anomaly detection. Neurocomputing 273
(2018), 535–546.  

<a href="#ref7" id="ref7">[7]</a> Hansheng Ren, Bixiong Xu, Yujing Wang, Chao Yi, Congrui Huang, Xiaoyu Kou, Tony Xing, Mao Yang, Jie Tong, and Qi Zhang. 2019. Time-series
anomaly detection service at microsoft. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 3009–3017.  

<a href="#ref8" id="ref8">[8]</a> Haowen Xu, Wenxiao Chen, Nengwen Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan Pei, Yang Feng, et al. 2018. Unsupervised
anomaly detection via variational auto-encoder for seasonal kpis in web applications. In World Wide Web Conference. 187–196.  

<a href="#ref9" id="ref9">[9]</a> Wenxiao Chen, Haowen Xu, Zeyan Li, Dan Pei, Jie Chen, Honglin Qiao, Yang Feng, and Zhaogang Wang. 2019. Unsupervised anomaly detection
for intricate kpis via adversarial training of vae. In IEEE INFOCOM 2019-IEEE Conference on Computer Communications. IEEE, 1891–1899.  

<a href="#ref10" id="ref10">[10]</a> Zeyan Li, Wenxiao Chen, and Dan Pei. 2018. Robust and unsupervised kpi anomaly detection based on conditional variational autoencoder. In
International Performance Computing and Communications Conference (IPCCC). IEEE, 1–9.  

<a href="#ref11" id="ref11">[11]</a> Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam Shroff. 2016. LSTM-based encoder-decoder
for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148 (2016).

<a href="#ref12" id="ref12">[12]</a> Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. 2018. Detecting spacecraft anomalies using lstms
and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining.
387–395.
	
<a href="#ref13" id="ref13">[13]</a> Sucheta Chauhan and Lovekesh Vig. 2015. Anomaly detection in ECG time signals via deep long short-term memory networks. In 2015 IEEE
International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 1–7.
	
<a href="#ref14" id="ref14">[14]</a> Jonathan Goh, Sridhar Adepu, Marcus Tan, and Zi Shan Lee. 2017. Anomaly detection in cyber physical systems using recurrent neural networks.
In 2017 IEEE 18th International Symposium on High Assurance Systems Engineering (HASE). IEEE, 140–145.
	
<a href="#ref15" id="ref15">[15]</a> Nan Ding, HaoXuan Ma, Huanbo Gao, YanHua Ma, and GuoZhen Tan. 2019. Real-time anomaly detection based on long short-Term memory and
Gaussian Mixture Model. Computers & Electrical Engineering 79 (2019), 106458.
	
<a href="#ref16" id="ref16">[16]</a> Lifeng Shen, Zhuocong Li, and James Kwok. 2020. Timeseries anomaly detection using temporal hierarchical one-class network. Advances in
Neural Information Processing Systems 33 (2020), 13016–13026.
	
<a href="#ref17" id="ref17">[17]</a> Wentai Wu, Ligang He, Weiwei Lin, Yi Su, Yuhua Cui, Carsten Maple, and Stephen A Jarvis. 2020. Developing an unsupervised real-time anomaly
detection scheme for time series with multi-seasonality. IEEE Transactions on Knowledge and Data Engineering (2020).
	
<a href="#ref18" id="ref18">[18]</a> Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. 2018. DeepAnT: A deep learning approach for unsupervised anomaly
detection in time series. Ieee Access 7 (2018), 1991–2005.
	
<a href="#ref19" id="ref19">[19]</a> Yangdong He and Jiabao Zhao. 2019. Temporal convolutional networks for anomaly detection in time series. In Journal of Physics: Conference
Series, Vol. 1213. IOP Publishing, 042050.
	
<a href="#ref20" id="ref20">[20]</a> Ailin Deng and Bryan Hooi. 2021. Graph neural network-based anomaly detection in multivariate time series. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 4027–4035.
	
<a href="#ref21" id="ref21">[21]</a> Zekai Chen, Dingshuo Chen, Xiao Zhang, Zixuan Yuan, and Xiuzhen Cheng. 2021. Learning graph structures with transformer for multivariate
time series anomaly detection in iot. IEEE Internet of Things Journal (2021).
	
<a href="#ref22" id="ref22">[22]</a> Enyan Dai and Jie Chen. 2022. Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series. arXiv preprint arXiv:2202.07857
(2022).
	
<a href="#ref23" id="ref23">[23]</a> Nan Ding, Huanbo Gao, Hongyu Bu, Haoxuan Ma, and Huaiwei Si. 2018. Multivariate-time-series-driven real-time anomaly detection based on
bayesian network. Sensors 18, 10 (2018), 3367.
	
<a href="#ref24" id="ref24">[24]</a> Huan Song, Deepta Rajan, Jayaraman Thiagarajan, and Andreas Spanias. 2018. Attend and diagnose: Clinical time series analysis using attention
models. In Proceedings of the AAAI conference on artificial intelligence, Vol. 32.
	
<a href="#ref25" id="ref25">[25]</a> Mayu Sakurada and Takehisa Yairi. 2014. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Workshop on Machine
Learning for Sensory Data Analysis. 4–11.
	
<a href="#ref26" id="ref26">[26]</a> Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and Haifeng Chen. 2018. Deep autoencoding gaussian
mixture model for unsupervised anomaly detection. In International conference on learning representations.
	
<a href="#ref27" id="ref27">[27]</a> Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V
Chawla. 2019. A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data. In Proceedings of the
AAAI conference on artificial intelligence, Vol. 33. 1409–1416.
	
<a href="#ref28" id="ref28">[28]</a> Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, and Maria A Zuluaga. 2020. Usad: Unsupervised anomaly detection on
multivariate time series. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 3395–3404.
	
<a href="#ref29" id="ref29">[29]</a> Adam Goodge, Bryan Hooi, See-Kiong Ng, and Wee Siong Ng. 2020. Robustness of Autoencoders for Anomaly Detection Under Adversarial
Impact.. In IJCAI. 1244–1250.
	
<a href="#ref30" id="ref30">[30]</a> Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. 2021. Practical approach to asynchronous multivariate time series anomaly detection
and localization. In ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2485–2494.
	
<a href="#ref31" id="ref31">[31]</a> David Campos, Tung Kieu, Chenjuan Guo, Feiteng Huang, Kai Zheng, Bin Yang, and Christian S Jensen. 2021. Unsupervised Time Series Outlier
Detection with Diversity-Driven Convolutional Ensembles–Extended Version. arXiv preprint arXiv:2111.11108 (2021).
	
<a href="#ref32" id="ref32">[32]</a> Yuxin Zhang, Jindong Wang, Yiqiang Chen, Han Yu, and Tao Qin. 2022. Adaptive memory networks with self-supervised learning for unsupervised
anomaly detection. IEEE Transactions on Knowledge and Data Engineering (2022).
	
<a href="#ref33" id="ref33">[33]</a> Daehyung Park, Yuuna Hoshi, and Charles C Kemp. 2018. A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational
autoencoder. IEEE Robotics and Automation Letters 3, 3 (2018), 1544–1551.
	
<a href="#ref34" id="ref34">[34]</a> Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. 2019. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2828–2837.
	
<a href="#ref35" id="ref35">[35]</a> Maximilian Sölch, Justin Bayer, Marvin Ludersdorfer, and Patrick van der Smagt. 2016. Variational inference for on-line anomaly detection in
high-dimensional time series. arXiv preprint arXiv:1602.07109 (2016).
	
<a href="#ref36" id="ref36">[36]</a> Yifan Guo, Weixian Liao, Qianlong Wang, Lixing Yu, Tianxi Ji, and Pan Li. 2018. Multidimensional time series anomaly detection: A gru-based
gaussian mixture variational autoencoder approach. In Asian Conference on Machine Learning. PMLR, 97–112.
	
<a href="#ref37" id="ref37">[37]</a> Longyuan Li, Junchi Yan, Haiyang Wang, and Yaohui Jin. 2020. Anomaly detection of time series with smoothness-inducing sequential variational
auto-encoder. IEEE transactions on neural networks and learning systems 32, 3 (2020), 1177–1191.
	
<a href="#ref38" id="ref38">[38]</a> Zijian Niu, Ke Yu, and Xiaofei Wu. 2020. LSTM-based VAE-GAN for time-series anomaly detection. Sensors 20, 13 (2020), 3738.
	
<a href="#ref39" id="ref39">[39]</a> Chunkai Zhang, Shaocong Li, Hongye Zhang, and Yingyang Chen. 2019. VELC: A new variational autoencoder based model for time series
anomaly detection. arXiv preprint arXiv:1907.01702 (2019).
	
<a href="#ref40" id="ref40">[40]</a> Zilong He, Pengfei Chen, Xiaoyun Li, Yongfeng Wang, Guangba Yu, Cailin Chen, Xinrui Li, and Zibin Zheng. 2020. A spatiotemporal deep learning
approach for unsupervised anomaly detection in cloud systems. IEEE Transactions on Neural Networks and Learning Systems (2020).
	
<a href="#ref41" id="ref41">[41]</a> Run-Qing Chen, Guang-Hui Shi, Wan-Lei Zhao, and Chang-Hui Liang. 2021. A joint model for IT operation series prediction and anomaly
detection. Neurocomputing 448 (2021), 130–139.
	
<a href="#ref42" id="ref42">[42]</a> Zhihan Li, Youjian Zhao, Jiaqi Han, Ya Su, Rui Jiao, Xidao Wen, and Dan Pei. 2021. Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding. In ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 3220–3230.
	
<a href="#ref43" id="ref43">[43]</a> Xixuan Wang, Dechang Pi, Xiangyan Zhang, Hao Liu, and Chang Guo. 2022. Variational transformer-based anomaly detection approach for
multivariate time series. Measurement 191 (2022), 110791.

<a href="#ref44" id="ref44">[44]</a> Longyuan Li, Junchi Yan, Qingsong Wen, Yaohui Jin, and Xiaokang Yang. 2022. Learning Robust Deep State Space for Unsupervised Anomaly
Detection in Contaminated Time-Series. IEEE Transactions on Knowledge and Data Engineering (2022).
	
<a href="#ref45" id="ref45">[45]</a> Dan Li, Dacheng Chen, Baihong Jin, Lei Shi, Jonathan Goh, and See-Kiong Ng. 2019. MAD-GAN: Multivariate anomaly detection for time series
data with generative adversarial networks. In International conference on artificial neural networks. Springer, 703–716.
	
<a href="#ref46" id="ref46">[46]</a> Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. 2019. BeatGAN: Anomalous Rhythm Detection using Adversarially Generated
Time Series. In IJCAI. 4433–4439.
	
<a href="#ref47" id="ref47">[47]</a> Xuanhao Chen, Liwei Deng, Feiteng Huang, Chengwei Zhang, Zongquan Zhang, Yan Zhao, and Kai Zheng. 2021. Daemon: Unsupervised anomaly
detection and interpretation for multivariate time series. In 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 2225–2230.
	
<a href="#ref48" id="ref48">[48]</a> Bowen Du, Xuanxuan Sun, Junchen Ye, Ke Cheng, Jingyuan Wang, and Leilei Sun. 2021. GAN-Based Anomaly Detection for Multivariate Time
Series Using Polluted Training Set. IEEE Transactions on Knowledge and Data Engineering (2021).
	
<a href="#ref49" id="ref49">[49]</a> Yifan Li, Xiaoyan Peng, Jia Zhang, Zhiyong Li, and Ming Wen. 2021. DCT-GAN: Dilated Convolutional Transformer-based GAN for Time Series
Anomaly Detection. IEEE Transactions on Knowledge and Data Engineering (2021).
	
<a href="#ref50" id="ref50">[50]</a> Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. 2021. Anomaly transformer: Time series anomaly detection with association discrepancy.
arXiv preprint arXiv:2110.02642 (2021).
	
<a href="#ref51" id="ref51">[51]</a> Shreshth Tuli, Giuliano Casale, and Nicholas R Jennings. 2022. TranAD: Deep transformer networks for anomaly detection in multivariate time
series data. arXiv preprint arXiv:2201.07284 (2022).
	
<a href="#ref52" id="ref52">[52]</a> Yuxin Zhang, Yiqiang Chen, Jindong Wang, and Zhiwen Pan. 2021. Unsupervised deep anomaly detection for multi-sensor time-series signals.
IEEE Transactions on Knowledge and Data Engineering (2021).
	
<a href="#ref53" id="ref53">[53]</a> Cheng Feng and Pengwei Tian. 2021. Time series anomaly detection for cyber-physical systems via neural system identification and bayesian
filtering. In ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2858–2867.
	
<a href="#ref54" id="ref54">[54]</a> Md Abul Bashar and Richi Nayak. 2020. TAnoGAN: Time series anomaly detection with generative adversarial networks. In 2020 IEEE Symposium
Series on Computational Intelligence (SSCI). IEEE, 1778–1785.
	
<a href="#ref55" id="ref55">[55]</a> Hang Zhao, Yujing Wang, Juanyong Duan, Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. 2020. Multivariate
time-series anomaly detection via graph attention network. In 2020 IEEE International Conference on Data Mining (ICDM). IEEE, 841–850.
	
<a href="#ref56" id="ref56">[56]</a> Siho Han and Simon S Woo. 2022. Learning Sparse Latent Graph Representations for Anomaly Detection in Multivariate Time Series. In Proceedings
of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2977–2986.

<a href="#ref57" id="ref57">[57]</a> Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. 2023. TimesNet: Temporal 2D-Variation Modeling for General
Time Series Analysis. In ICLR.

<a href="#ref58" id="ref58">[58]</a> Kwei-Herng Lai, Lan Wang, Huiyuan Chen, Kaixiong Zhou, Fei Wang, Hao Yang, and Xia Hu. 2023. Context-aware domain adaptation for time
series anomaly detection. In SDM. SIAM, 676–684.

<a href="#ref59" id="ref59">[59]</a> Youngeun Nam, Susik Yoon, Yooju Shin, Minyoung Bae, Hwanjun Song, Jae-Gil Lee, and Byung Suk Lee. 2024. Breaking the Time-Frequency
Granularity Discrepancy in Time-Series Anomaly Detection. In Proceedings of the ACM on Web Conference 2024. 4204–4215.

<a href="#ref60" id="ref60">[60]</a> Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. 2022. Ts2vec: Towards universal
representation of time series. In AAAI, Vol. 36. 8980–8987.

<a href="#ref61" id="ref61">[61]</a> Xiang Zhang, Ziyuan Zhao, Theodoros Tsiligkaridis, and Marinka Zitnik. 2022. Self-supervised contrastive pre-training for time series via
time-frequency consistency. NeurIPS 35 (2022), 3988–4003.

<a href="#ref62" id="ref62">[62]</a> Zahra Zamanzadeh Darban, Geoffrey I Webb, Shirui Pan, Charu C Aggarwal, and Mahsa Salehi. 2023. CARLA: Self-supervised contrastive representation learning for time series anomaly detection. Pattern Recognition (2024).

<a href="#ref63" id="ref63">[63]</a> Zahra Zamanzadeh Darban, Geoffrey I Webb, and Mahsa Salehi. 2024. DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection
in Multivariate Time Series. arXiv preprint arXiv:2404.11269 (2024).



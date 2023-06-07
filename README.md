
# Deep Learning for Time Series Anomaly Detection (Models and Datasets)
*Time-Series Anomaly Detection Comprehensive Benchmark*

This repository updates the comprehensive list of classic and state-of-the-art methods and datasets for Anomaly Decetion in Time-Series. This is part of an onging research at Time Series Analytics Lab, Monash University.

If you use this repository in your works, please cite the main article:

[-] Zamanzadeh Darban, Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Salehi, M. (2022). Deep Learning for Time Series Anomaly Detection: A Survey. arXiv e-prints. doi:10.48550/ARXIV.2211.05244 [[arXiv](https://arxiv.org/abs/2211.05244)]

	@ARTICLE{2022arXiv221105244Z,
		author = {{Zamanzadeh Darban}, Zahra and {Webb}, Geoffrey I. and {Pan}, Shirui and {Aggarwal}, Charu C. and {Salehi}, Mahsa},
		title = {Deep Learning for Time Series Anomaly Detection: A Survey},
		journal = {arXiv e-prints},
		year = 2022,
		month = Nov,
		eid = {arXiv:2211.05244},
		eprint = {2211.05244},
		url = {https://arxiv.org/abs/2211.05244},
		doi = {10.48550/ARXIV.2211.05244},
	}

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

| A<sup>[1](#Approach)</sup> | MA<sup>[2](#Main)</sup> | Model | Su/Un<sup>[3](#Su)</sup> | Input | P/S<sup>[4](#point)</sup> | Code |
| --- | --- | --- | --- | --- | --- | --- |
| **Forecasting** | RNN | LSTM-AD <a href="#ref1" id="ref1">[1]</a> | Un | P | Point |
| **Forecasting** | RNN | LSTM RNN <a href="#ref2" id="ref2">[2]</a> | Semi | P | Subseq |
| **Forecasting** | RNN | LSTM-based <a href="#ref3" id="ref3">[3]</a> | Un | W | - |
| **Forecasting** | RNN | TCQSA <a href="#ref4" id="ref4">[4]</a> | Su | P | - |
| **Forecasting** | HTM | Numenta HTM <a href="#ref5" id="ref5">[5]</a> | Un | - | - |
| **Forecasting** | HTM | Multi HTM <a href="#ref6" id="ref6">[6]</a> | Un | - | - |
| **Forecasting** | CNN | SR-CNN <a href="#ref7" id="ref7">[7]</a> | Un | W | Point + Subseq |
| **Reconstruction** | VAE | Donut <a href="#ref8" id="ref8">[8]</a> | Un | W | Subseq |
| **Reconstruction** | VAE | Buzz <a href="#ref9" id="ref9">[9]</a> | Un | W | Subseq |
| **Reconstruction** | VAE | Bagel <a href="#ref10" id="ref10">[10]</a> | Un | W | Subseq |
| **Reconstruction** | AE | EncDec-AD <a href="#ref11" id="ref11">[11]</a> | Semi | W | Point |
  

## Multivariate Deep Anomaly Detection Models in Time Series
	
| A<sup>[1](#Approach)</sup>  | MA<sup>[2](#Main)</sup> | Model | T/S$^3$ | Su/Un<sup>[4](#Su)</sup>  | Input | Int$^5$ | P/S<sup>[6](#point)</sup> | Stc$^7$ | Inc$^8$ |
|-------|--------|-------|---------|-----------|-------|----------|---------|---------|---------|
| Forecasting | RNN | LSTM-NDT <a href="#ref12" id="ref12">[12]</a> | T | Un | W | &#10003; | Subseq | | |
| Forecasting | RNN | DeepLSTM <a href="#ref13" id="ref13">[13]</a> | T | Semi | P | | Point | | |
| Forecasting | RNN | LSTM-PRED <a href="#ref14" id="ref14">[14]</a> | T | Un | W | &#10003; | - | | |
| Forecasting | RNN | LGMAD <a href="#ref15" id="ref15">[15]</a> | T | Semi | P | | Point | | |
| Forecasting | RNN | THOC <a href="#ref16" id="ref16">[16]</a> | T | Self | W | | Subseq | | |
| Forecasting | RNN | AD-LTI <a href="#ref17" id="ref17">[17]</a> | T | Un | P | | Point (frame) | | |
| Forecasting | CNN | DeepAnt <a href="#ref18" id="ref18">[18]</a> | T | Un | W | | Point + Subseq | | |
| Forecasting | CNN | TCN-ms <a href="#ref19" id="ref19">[19]</a> | T | Semi | W | | Subseq | | |
| Forecasting | GNN | GDN <a href="#ref20" id="ref20">[20]</a> | S | Un | W | &#10003; | - | | |
| Forecasting | GNN | GTA* <a href="#ref21" id="ref21">[21]</a> | ST | Semi | - | | - | | |
| Forecasting | GNN | GANF <a href="#ref22" id="ref22">[22]</a> | ST | Un | W | | | | |
| Forecasting | HTM | RADM <a href="#ref23" id="ref23">[23]</a> | T | Un | W | | - | | |
| Forecasting | Transformer | SAND <a href="#ref24" id="ref24">[24]</a> | T | Semi | W | | - | | |
| Forecasting | Transformer | GTA* <a href="#ref21" id="ref21">[21]</a> | ST | Semi | - | | - | | |
| Reconstruction | AE | AE/DAE <a href="#ref25" id="ref25">[25]</a> | T | Semi | P | | Point | | |
| Reconstruction | AE | DAGMM <a href="#ref26" id="ref26">[26]</a> | S | Un | P | | Point | &#10003; | |
| Reconstruction | AE | MSCRED <a href="#ref27" id="ref27">[27]</a> | ST | Un | W | &#10003; | Subseq | | |
| Reconstruction | AE | USAD <a href="#ref28" id="ref28">[28]</a> | T | Un | W | | Point | | |
| Reconstruction | AE | APAE <a href="#ref29" id="ref29">[29]</a> | T | Un | W | | - | | |
| Reconstruction | AE | RANSynCoders <a href="#ref30" id="ref30">[30]</a> | ST | Un | P | &#10003; | Point | | &#10003; |
| Reconstruction | AE | CAE-Ensemble <a href="#ref31" id="ref31">[31]</a> | T | Un | W | | Subseq | | |
| Reconstruction | AE | AMSL <a href="#ref32" id="ref32">[32]</a> | T | Self | W | | - | | |
| Reconstruction | VAE | LSTM-VAE <a href="#ref33" id="ref33">[33]</a> | T | Semi | P | | - | | |
| Reconstruction | VAE | OmniAnomaly <a href="#ref34" id="ref34">[34]</a> | T | Un | W | &#10003; | Point + Subseq | &#10003; | |
| Reconstruction | VAE | STORN <a href="#ref35" id="ref35">[35]</a> | ST | Un | P | | Point | | |
| Reconstruction | VAE | GGM-VAE <a href="#ref36" id="ref36">[36]</a> | T | Un | W | | Subseq | | |
| Reconstruction | VAE | SISVAE <a href="#ref37" id="ref37">[37]</a> | T | Un | W | | Point | | |
| Reconstruction | VAE | VAE-GAN <a href="#ref38" id="ref38">[38]</a> | T | Semi | W | | Point | | |
| Reconstruction | VAE | VELC <a href="#ref39" id="ref39">[39]</a> | T | Un | - | | - | | |
| Reconstruction | VAE | TopoMAD <a href="#ref40" id="ref40">[40]</a> | ST | Un | W | | Subseq | &#10003; | |
| Reconstruction | VAE | PAD <a href="#ref41" id="ref41">[41]</a> | T | Un | W | | Subseq | | |
| Reconstruction | VAE | InterFusion <a href="#ref42" id="ref42">[42]</a> | ST | Un | W | &#10003; | Subseq | | |
| Reconstruction | VAE | MT-RVAE* <a href="#ref43" id="ref43">[43]</a> | ST | Un | W | | - | | |
| Reconstruction | VAE | RDSMM <a href="#ref44" id="ref44">[44]</a> | T | Un | W | | Point + Subseq | &#10003; | |
| Reconstruction | GAN | MAD-GAN <a href="#ref45" id="ref45">[45]</a> | ST | Un | W | | Subseq | | |
| Reconstruction | GAN | BeatGAN <a href="#ref46" id="ref46">[46]</a> | T | Un | W | | Subseq | | |
| Reconstruction | GAN | DAEMON <a href="#ref47" id="ref47">[47]</a> | T | Un | W | &#10003; | Subseq | | |
| Reconstruction | GAN | FGANomaly <a href="#ref48" id="ref48">[48]</a> | T | Un | W | | Point + Subseq | | |
| Reconstruction | GAN | DCT-GAN* <a href="#ref49" id="ref49">[49]</a> | T | Un | W | | - | | |
| Reconstruction | Transformer | Anomaly Transformer <a href="#ref50" id="ref50">[50]</a> | T | Un | W | | Subseq | | |
| Reconstruction | Transformer | TranAD <a href="#ref51" id="ref51">[51]</a> | T | Un | W | &#10003; | Subseq | | |
| Reconstruction | Transformer | DCT-GAN* <a href="#ref49" id="ref49">[49]</a> | T | Un | W | | - | | |
| Reconstruction | Transformer | MT-RVAE* <a href="#ref43" id="ref43">[43]</a> | ST | Un | W | | - | | |
| Hybrid | AE | CAE-M <a href="#ref52" id="ref52">[55]</a> | ST | Un | W | | Subseq | | |
| Hybrid | AE | NSIBF* <a href="#ref53" id="ref53">[53]</a> | T | Un | W | | Subseq | | |
| Hybrid | RNN | NSIBF* <a href="#ref53" id="ref53">[53]</a> | T | Un | W | | Subseq | | |
| Hybrid | RNN | TAnoGAN <a href="#ref54" id="ref54">[54]</a> | T | Un | W | | Subseq | | |
| Hybrid | GNN | MTAD-GAT <a href="#ref55" id="ref55">[55]</a> | ST | Self | W | &#10003; | Subseq | | |
| Hybrid | GNN | FuSAGNet <a href="#ref56" id="ref56">[56]</a> | ST | Semi | W | | Subseq | | |

---
<sub><a name="Approach">1</a>: Approach. </sub>

<sub><a name="Main">2</a>: Main Approach. </sub>

<sub><a name="Su">3</a>:

<sub><a name="Su">4</a>: Supervised/Unsupervised | Values: [Su: Supervised, Un: Unsupervised, Semi: Semi-supervised, Self: Self-supervised]. </sub>

<sub><a name="point">6</a>: Point/Sub-sequence. </sub>
	
	
## References
<a id="ref1" href="#ref1">[1]</a> Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, Puneet Agarwal, et al . 2015. Long short term memory networks for anomaly detection in time
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








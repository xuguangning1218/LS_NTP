#  Near-surface Temperature Prediction

Official source code for paper 《LS-NTP: Unifying Long- and Short-range Spatial Correlations for Near-surface Temperature Prediction》
### Overall Architecture of LS-NTP
![image](https://github.com/xuguangning1218/LS_NTP/blob/master/figure/network.png)

### Environment Installation
```
conda env create -f LS_NTP.yaml
```  

### Data Preparation 
* Download the required temperature dataset from ERA5 offical site through [here](<https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview> "here") and the required temperature dataset from NCEP offical site through [here](<https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html>  "here"). 
* Or you can download the preprocessing data from my google drive through [here](<https://drive.google.com/drive/folders/1AWxRfWDtAOejI9y7PK1V0f92Gjlchhfi?usp=sharing> "here").

###  Reproducibility
We provide one of the five runs best-validated models for both ERA5 and NCEP datasets in [here](<https://drive.google.com/drive/folders/1wp7odEkxfLxLeHAH36q3y2TUzMk1PWt6?usp=sharing>  "here").  You can reproduce the result reported in the paper using this best-validated model.


###  Source Files Description

```
-- data # dataset folder
	-- era5 # the ERA5 dataset
	-- ncep # the NCEP dataset
-- dataprovider # data reader and normalizer
	-- era5.py # dataloader in train, validate, test for ERA5
	-- ncep.py # dataloader in train, validate, test for NCEP
	-- normalizer.py # data normalizer, including std, maxmin
-- figure # figure provider
	-- network.png # architecture of LS-NTP model 
-- model # proposed model
	-- lsconv.py # the proposed LS-Conv
	-- lsconvlstm.py # the ConvLSTM with LS-Conv
	-- model.py # model loader, saver, procedure of train, validate, and test
	-- network.py # the LS-NTP
-- save # model save path
	-- era5 # best model on ERA5 (one of five runs)
	-- ncep # best model on NCEP (one of five runs)
LS_NTP.yaml # conda environment for the project
lsntp_era5.config # model configure for ERA5
lsntp_ncep.config # model configure for NCEP
Run_era5.ipynb # jupyter visualized code for the whole procedure on ERA5
Run_ncep.ipynb # jupyter visualized code for the whole procedure on NCEP
```

### Run

When the conda environment and datasets are ready, you can train or reproduce our result by running file `Run_era5.ipynb` or `Run_ncep.ipynb`.

### Citation
If you find this code or idea useful, please cite our work:
```bib
@article{XU2022242,
  title = {LS-NTP: Unifying long- and short-range spatial correlations for Near-surface Temperature Prediction},
  journal = {Neural Networks},
  volume = {155},
  pages = {242-257},
  year = {2022},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2022.07.022},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608022002787},
  author = {Guangning Xu and Xutao Li and Shanshan Feng and Yunming Ye and Zhihua Tu and Kenghong Lin and Zhichao Huang},
}
```

# 📡 CSIRO Data Sources & Search Log

This document tracks the specific raw data files (.sf) used to build and calibrate this pipeline. 
If you want to reproduce the results, these are the exact observations that were pulled from the Parkes Radio Telescope archives.

#### 🔍 The Search Strategy
To find these files on the CSIRO Data Access Portal, use the following parameters to filter through the massive HTRU survey:
* Search Term: "Parkes observations for project P630 semester "
* Direct Link: [Click here for Search Results](https://data.csiro.au/search/keyword?q=Parkes%20observations%20for%20project%20P630%20semester%20&p=3&rpp=25&sb=TITLE)
* Sorting: By Title 

#### 🎯 The Calibration Datasets
These are the files where there were successfully detected pulsars and used the data to train the Neural Network.
1. The Millisecond Pulsar (PSR J0437-4715)
* Collection: [Parkes observations for project P630 semester 2009APRS_04](https://data.csiro.au/collection/csiro:49517?q=Parkes%20observations%20for%20project%20P630%20semester%20&p=1&rpp=25&sb=TITLE&_st=keyword&_str=122&_si=18)
* Target File: bpsr081121_171829_beam04.sf

2. The Normal Pulsar (PSR J1048-5832)
* Collection: [Parkes observations for project P630 semester 2010APRS_04](https://data.csiro.au/collection/csiro:53169?q=Parkes%20observations%20for%20project%20P630%20semester%20&p=3&rpp=25&sb=TITLE&_st=keyword&_str=143&_si=67)
* Target File: bpsr100613_052818_beam08.sf

#### 🏅 Honorary Mention
This file was the initial file to process which it might just contain a pulsar. While not used for the final classification model, it was crucial for calibrating the RFI masking and High-DM logic.
* Collection: [Parkes observations for project P630 semester 2008OCTS_04](https://data.csiro.au/collection/csiro:56626?q=Parkes%20observations%20for%20Project%20P630%20semester%20&p=1&rpp=25&sb=TITLE&_st=keyword&_str=132&_si=5)
* Target File: bpsr090603_162807_beam13.sf

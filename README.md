The COVID-19 forecasting model used here is simplified from the one studied the paper [https://www.nature.com/articles/s41598-022-11693-9] that I am a co-author. The main idea is to use an auto-variational encoder architecture to learn from the historical data the three coefficients (contact rate, recovered rate, and death rate) of the SIRD equations modeling the disease in long time. Here I simplified one LSTM layer but add a convolutional layer to capture more features from the data. To run the forecasting please follow the steps below.

1. Install Anaconda environment using yalm file for running on CPU:
   ```
   cd covid-19-forecasting
   conda update conda
   conda env create -f environment.yml
   ```
   If you aim to run the forecasting on GPU, you need to install ```mxnet``` with CUDA support (CUDA version is 11.2 or lower otherwise you need to downgrade your current CUDA version to those versions or use CPU instead). On Linux OS, if CUDA version is 11.2 then run:
   ```
   pip install mxnet-cu112
   ```
   On other operating systems, please refer to [https://mxnet.apache.org/versions/1.9.1/get_started?] for more information.
3. Run ```main.py``` with flag ```--train``` if the model needs to be re-trained. Otherwise, just simply run:
   ```
   python main.py
   ```
3. Example of COVID-19 forecasting in Italy:
   ![Alt text](img/infected.png)![Alt text](img/deceased.png)![Alt text](img/recovered.png)
5. Modify the configuration file "configs/config.yaml" if you want to forecast the COVID-19 disease in other countries or for training settings. You can add more layers to the model in ```models/model.py``` to make the forecasting more precise.

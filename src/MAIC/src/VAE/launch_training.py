#!/usr/bin/env python3.7

import numpy as np
from pathlib import Path
from AutoEncoder import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("Device =", device)

data_id = ""
model_id = "AutoEncoder"
network = AutoEncoder()
    
def main():
 
    max_epochs = 51
    batch_size = 128
 
    # Make a directory for the model to be trained
    Path(network.SAVE_PATH + "/" + model_id + "/").mkdir(parents=True, exist_ok=True)

    # Save a copy of the data range of the data at the model location
    np.save(network.SAVE_PATH + "/" + model_id + "/data_range" + model_id, np.array([0,1]))
    network.train_net(network, model_id, max_epochs, batch_size) 


  
   
if __name__ == '__main__':
    main()
    

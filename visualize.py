import numpy as np
import matplotlib.pyplot as plt
import os

class Visualize:
    def __init__(self,epoch,visualize_dir):
        self.epoch = epoch
        self.visualize_dir = visualize_dir

    def plot_feature_map(self,epoch,features):
        b,d,h,w = features.shape
        print(b,d,h,w)
        for i,feature in enumerate(features):
            fig, ax = plt.subplots(nros=1,ncols=d/1,figsize = (d/1,1))
            for j,f in enumerate(feature):
                ax[j].imshow(f)
            plt.savefig(os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))
            print("save in : ",os.path.join(self.visualize_dir,str(epoch)+"_"+str(i)+".png"))
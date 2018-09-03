from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import pandas as pd
import pylab
import numpy as np
from sklearn.metrics import mean_squared_error

image = imread('D:\\Dev\\ML\\datasets\\coursera\\parrots.jpg')
img_float = img_as_float(image)

# r,g,b = img_float[:,:, 0], img_float[:,:, 1], img_float[:,:, 2]
X = pd.DataFrame(np.column_stack((img_float[:,:, 0].reshape(-1,1),
                         img_float[:,:, 1].reshape(-1,1),
                         img_float[:,:, 2].reshape(-1,1))))

def km_train(n):
    km = KMeans(init='k-means++',n_clusters=n ,random_state=241)
    return km.fit(X)

def main(n_clusters):
    km = km_train(n_clusters)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X.index.values
    cluster_map['cluster'] = km.labels_
    median_arr = get_median(len(km.cluster_centers_), cluster_map)
    median_MSE = get_MSE(median_arr,cluster_map)
    mean_MSE = get_MSE(km.cluster_centers_,cluster_map)
    median_PSNR = PCNR(median_MSE)
    mean_PSNR = PCNR(mean_MSE)

    return np.column_stack((mean_PSNR,median_PSNR))
    # return "PSNR_mean: "+str(mean_PSNR.mean())+"" \
    #                                     " PSNR_median: "+str(median_PSNR.mean())



def get_median(labels_count,cluster_map):
    median_arr = []
    for i in range(labels_count):
        cluster_val = np.array(X.take(cluster_map[cluster_map.cluster == i]['data_index']))
        median_arr.append(np.median(cluster_val))
    return np.array(median_arr)

def PCNR(mse):
    pcnr_list = []
    for i in range(len(mse)):
        pcnr_list.append(10 * np.log10(1 / mse[i]))

    return np.array(pcnr_list)


def get_MSE(x_true, x_pred):
    mse_arr = []
    for i in range(len(x_true)):
        cluster_val = np.array(X.take(x_pred[x_pred.cluster == i]['data_index']))
        cls_bst = np.broadcast_to(x_true[i],(len(cluster_val),3))
        mse_arr.append(mean_squared_error(cls_bst,cluster_val))
    return np.array(mse_arr)
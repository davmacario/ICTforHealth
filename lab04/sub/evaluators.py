import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.cluster as sk
import sklearn.metrics as metrics

def evalAccuracy(col_predict, col_label):
    """
    evalAccuracy
    ----------------------------------------------------------
    Find accuracy given estimated classes and actual classes
    ----------------------------------------------------------
    Parameters:
    - col_predict: array containing the predicted classes
    - col_label: array containing the actual labels
    ----------------------------------------------------------
    Outputs:
    - accuracy (float)
    """
    if (len(col_predict.shape) > 1 or len(col_label.shape) > 1):
        if (not all(col_predict.shape[1:] == 1) or not all(col_label.shape[1:] == 1)):
            raise ValueError("The passed arrays are not 1D")
    
    n_elem = col_predict.shape[0]
    if (col_label.shape[0] != n_elem):
        raise ValueError("The two arrays don't have the same length!")

    n_classes = len(np.unique(col_label))
    tmp_sum = np.zeros((n_classes,))
    tmp_tot = np.zeros((n_classes,))
    
    for i in range(n_elem):
        tmp_tot[int(col_label[i])-1] += 1
        if col_label[i] == col_predict[i]:
            tmp_sum[int(col_label[i])-1] += 1

    accuracy = (1/19) * np.sum(tmp_sum/tmp_tot)

    return accuracy


def evalAccuracyClasses(col_predict, col_label):
    """
    Find accuracy value for each class
    ----------------------------------------------------------
    Parameters:
    - col_predict: array containing the predicted classes
      - col_label: array containing the actual labels
      ----------------------------------------------------------
      Outputs:
      - accuracy (ndarray)
    """
    if (len(col_predict.shape) > 1 or len(col_label.shape) > 1):
        if (not all(col_predict.shape[1:] == 1) or not all(col_label.shape[1:] == 1)):
            raise ValueError("The passed arrays are not 1D")
    
    n_elem = col_predict.shape[0]
    if (col_label.shape[0] != n_elem):
        raise ValueError("The two arrays don't have the same length!")
    
    n_classes = len(np.unique(col_label))
    tmp_sum = np.zeros((n_classes,))
    tmp_tot = np.zeros((n_classes,))

    for i in range(len(col_label)):
      tmp_tot[int(col_label[i])-1] += 1
      if col_label[i] == col_predict[i]:
            tmp_sum[int(col_label[i])-1] += 1
    
    acc_classes = tmp_sum/tmp_tot

    return acc_classes




def plotConfusionMatrix(col_predict, col_label, class_names, title='Confusion Matrix', save_img=False, img_path='./img/conf_matrix.png'):
    """
    plotConfusionMatrix
    ----------------------------------------------------------
    Plot the confusion matrix
    ----------------------------------------------------------
    """
    cm = metrics.confusion_matrix(col_label, col_predict)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    ax = sn.heatmap(cm_df, annot=True, cmap='BuPu')
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.title(title)
    if save_img:
      plt.savefig(img_path)
    plt.tight_layout()
    plt.show()

    return cm



def interCentroidDist(centroids, cent_names_axis, plot=False, save_img=False, img_path='img/inter_centroid_dist.png'):
    """
    interCentroidDist
    ----------------------------------------------------------
    Evaluate (and plot) the matrix containing as element i,j
    the distance between centroid i and centroid j
    ----------------------------------------------------------
    Parameters:
    - centroids: matrix containing the centroids as rows
    - cent_name_axis: list containing the names of the 
      centroids (used as labels of the plot)
    - plot: boolean flag for plotting
    - save_img: boolean flag for saving the image
    - img_path: path indicating where to store the image
    ----------------------------------------------------------
    Outputs:
    - d: square matrix containing in element i, j the distance 
      (norm) between element i and element j
    """
    
    # Element i, j is squared norm of dist between centroid i and j
    NAc = centroids.shape[0]
    
    d = np.zeros((NAc, NAc))

    for i in range(NAc):
        for j in range(i+1, NAc):       # Optimization - diagonal elements are 0 and matrix is symmetric
            d[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            d[j, i] = d[i, j]

    if plot:
        plt.matshow(d)
        plt.colorbar()
        plt.xticks(np.arange(NAc), cent_names_axis, rotation=90)
        plt.yticks(np.arange(NAc), cent_names_axis)
        plt.title('Between-centroids distance')
        if save_img:
            try:
                plt.savefig(img_path)
            except:
                plt.savefig('lab04/'+img_path)
        plt.show()

    return d

def minCentDist(centroids, cent_dist_matrix = None):
    """
    minCentDist
    ----------------------------------------------------------
    Used to find the minimum distance between each centroid 
    and all others.
    ----------------------------------------------------------
    Parameters:
    - centroids: matrix containing the centroids as rows
    - cent_dist_matrix: squared matrix of distances between 
      centroids
    ----------------------------------------------------------
    Outputs:
    - dmin: ndarray minimum distance between each centroid and 
      all others
    """
    if cent_dist_matrix is None:
        d = interCentroidDist(centroids, cent_names_axis = None)
    else:
        d = cent_dist_matrix

    NAc = centroids.shape[0]

    dd = d+np.eye(NAc)*(2*d.max())       # Set distance with itself (=0) to a large value (won't be the min)

    dmin = dd.min(axis=0)         # Find the minimum distance for each centroid

    return dmin

def avgDistCent(stdpoints):
    """
    avgDistCent
    ----------------------------------------------------------
    Evaluate the average distance between the centroids and 
    all points of the same cluster
    ----------------------------------------------------------
    Parameters:
    - stdpoints: matrix of standard deviations (w.r.t. 
      centroids) for each cluster
    ----------------------------------------------------------
    Outputs:
    - dpoints: ndarray containing for each cluster the average 
      distance
    """
    # Average distance between each centroid and its points 
    dpoints = np.sqrt(np.sum(stdpoints**2, axis=1))
    return dpoints

def centroidSeparationPlot(centroids, stdpoints, cent_names_axis, save_img=False, img_path='img/centroid_sep.png'):
    """
    centroidSeparionPlot
    ----------------------------------------------------------
    Plot the comparison between minimum inter-centroid 
    distance and average distance between centroids and each 
    element of the corresponding cluster
    ----------------------------------------------------------
    Parameters:
    - centroids: matrix containing the centroids as rows
    - stdpoints: matrix of standard deviations (w.r.t. 
      centroids) for each cluster
    - cent_name_axis: list containing the names of the 
      centroids (used as labels of the plot)
    - save_img: boolean flag for saving the image
    - img_path: path indicating where to store the image
    """
    NAc = centroids.shape[0]

    dmin = minCentDist(centroids)
    dpoints = avgDistCent(stdpoints)

    plt.figure()
    plt.plot(dmin, label='minimum centroid distance')
    plt.plot(dpoints, label='mean distance from points to centroid')
    plt.grid()
    plt.xticks(np.arange(NAc), cent_names_axis, rotation=90)
    plt.legend()
    plt.title('Centroid separation plot')
    plt.tight_layout()
    if save_img:
        plt.savefig(img_path)
    plt.show()


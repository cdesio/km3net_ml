import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt 

def ROC_curve_tot(y_true, y_pred, probs, title="Receiver operating characteristic curve"):
    n_classes = 2
    fpr = dict()
    tpr= dict()
    roc_auc=dict()
    Y_true = to_categorical(y_true)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:,i], probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true, y_pred)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr[0], tpr[0], color="darkorange",
             lw=lw, label="ROC curve downgoing (Area=%0.3f)" % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color="darkgreen",
             lw=lw, label="ROC curve upgoing (Area=%0.3f)" % roc_auc[1])

    plt.plot([0,1],[0,1], color="navy", lw=2.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    return
        
def energy_maps(energies, **kwargs):
    
    log_energies = np.log10(energies)
    
    E_maps = []
    binned, energy_bins = np.histogram(log_energies, bins=[log_energies.min(), 2.5, 3, 4, 5.5, log_energies.max() ])#,
                                  #range=(log_energies.min(), log_energies.max()))

    for i, E in enumerate(zip(energy_bins[:-1], energy_bins[1:])):
        lowE, highE = E
        E_maps.append((i, np.where(np.logical_and(log_energies>=lowE, log_energies<highE))[0]))
    
    return E_maps, energy_bins
    

def ROC_curve_energy(E_map_list, E_bins, y_true, y_pred, probs, var, title = "Receiver operating characteristic curve"):
    Y_true = to_categorical(y_true)
    n_classes = 2
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(10, 7))
    plt.grid()
    color_list = ["red", "blue", "green", "orange", "magenta", "brown", "black"]
    
    for em in range(len(E_map_list)):
        E_map = E_map_list[em][1]
    
        for i, ls in zip(range(n_classes-1), ['-', '--']):
            fpr[i], tpr[i], _ = roc_curve(Y_true[E_map][:,i], probs[E_map][:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            lw = 1.5
            
            plt.plot(fpr[i], tpr[i], color=color_list[em], linestyle=ls,
                     lw=lw, label="ROC curve {} E_bin:[{:.1f},{:.1f}] class {} (Area={:.3f})".format(var, E_bins[em], 
                                                                                                      E_bins[em+1], 
                                                                                                  i, roc_auc[i]))
            
            plt.plot([0,1],[0,1], color="navy", lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False positive rate", size=12)
            plt.ylabel("True positive rate", size=12)
            plt.title(title)
            plt.legend(loc="lower right")
            
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true[E_map], y_pred[E_map])
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
    return
    
    
    
def distance_maps(distances, **kwargs):
    
    d_maps = []
    binned, dist_bins = np.histogram(distances,bins=[6.13e-01, 170, 320, 450, 500, 8.106e+02])#, 
                                  #range=(log_energies.min(), log_energies.max()))
    for i, d in enumerate(zip(dist_bins[:-1],dist_bins[1:])):
        lowd, highd = d
        d_maps.append((i, np.where(np.logical_and(distances>=lowd, distances<highd))[0]))
    return d_maps, dist_bins
    
    
    
    
def ROC_curve_dist(E_map_list, E_bins, y_true, y_pred, probs, var, title = "Receiver operating characteristic curve"):
    Y_true = to_categorical(y_true)
    n_classes = 2
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(10, 7))
    plt.grid()
    color_list = ["red", "blue", "green", "orange", "magenta", "brown", "black"]
    
    for em in range(len(E_map_list)):
        E_map = E_map_list[em][1]
    
        for i, ls in zip(range(n_classes-1), ['-', '--']):
            fpr[i], tpr[i], _ = roc_curve(Y_true[E_map][:,i], probs[E_map][:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            lw = 1.5
            
            plt.plot(fpr[i], tpr[i], color=color_list[em], linestyle=ls,
                     lw=lw, label="ROC curve {} d_bin:[{},{}] class {} (Area={:.3f})".format(var, E_bins[em], 
                                                                                                      E_bins[em+1], 
                                                                                                  i, roc_auc[i]))
            
            plt.plot([0,1],[0,1], color="navy", lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel("False positive rate", size=12)
            plt.ylabel("True positive rate", size=12)
            plt.title(title)
            plt.legend(loc="lower right")
            
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true[E_map], y_pred[E_map])
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
    return
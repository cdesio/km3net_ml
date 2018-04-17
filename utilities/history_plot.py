import matplotlib
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt

def history_plot(history, name, flag):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.grid()
    plt.legend(['Training', 'Validation'])
    if flag=="save":
        plt.savefig("./plots/Loss_"+name+".png")
    elif (flag=="show"):
        plt.show()
    elif (flag=="both"):
        plt.savefig("./plots/Loss_"+name+".png")
        plt.show()
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(history.history['acc'])
    if 'val_acc' in history.history:
        plt.plot(history.history['val_acc'])
    plt.grid()
    plt.legend(['Training', 'Validation'], loc='lower right')
    if flag=="save":
        plt.savefig("./plots/Accuracy_"+name+".png")
    elif (flag=="show"):
        plt.show()
    elif (flag=="both"):
        plt.savefig("./plots/Accuracy_"+name+".png")
        plt.show()

COLORS = ['red', 'blue', 'green', 'orange', 'purple']
        
def history_plot_cv(history_cv,name, flag, K, N):
    """"""

    if N % 2 == 1:
        N += 1
    f, axarr = plt.subplots(N/2, 2, figsize=(15,15) if N>2 else (12,7), sharey=True)
    for it in range(N):
        row = it / 2
        col = it % 2
 
        fold_history = history_cv[K*it:K*(it+1)]  
        
        if N==2:

                        
            for i, history in enumerate(fold_history):
                axarr[it].set_xlabel('Epochs')
                axarr[it].set_ylabel('Loss',size=10)
                axarr[it].set_title('{} Iteration - Loss for {} Fold CV'.format(it+1, K))
                axarr[it].plot(history.history['loss'], label='Training Loss {}'.format(i+1),
                                 color=COLORS[i])
                axarr[it].plot(history.history['val_loss'], label='Validation Loss {}'.format(i+1),
                                 color=COLORS[i], linestyle='--')
                axarr[it].grid()
                axarr[it].legend(loc="upper right", prop={"size":10})
                
  
        else:
      
            axarr[row, col].set_xlabel('Epochs')
            axarr[row, col].set_ylabel('Loss')
            axarr[row, col].set_title('{} Iteration - Loss for {} Fold CV'.format(it+1, K))
        
            for i, history in enumerate(fold_history):
                axarr[row, col].plot(history.history['loss'], label='Training Loss {}'.format(i+1),
                                 color=COLORS[i])
                axarr[row, col].plot(history.history['val_loss'], label='Validation Loss {}'.format(i+1),
                                 color=COLORS[i], linestyle='--')
                axarr[row, col].grid()
                axarr[row, col].legend(loc="upper right", prop={"size":7})
                
        
    f.subplots_adjust(hspace=0.5)
    
    if flag=="save":
        plt.savefig("./plots/Loss_cv_"+name+".png")
    elif (flag=="show"):
        plt.show()
    elif (flag=="both"):
        plt.savefig("./plots/Loss_cv_"+name+".png")
        plt.show()
        
    f, axarr = plt.subplots(N/2, 2, figsize=(15,15) if N>2 else (12,7),sharey=True)
    for it in range(N):
        row = it / 2
        col = it % 2
        
        fold_history = history_cv[K*it:K*(it+1)]  
        if N==2:
            for i, history in enumerate(fold_history):
                axarr[it].set_xlabel('Epochs')
                axarr[it].set_ylabel('Accuracy')
                axarr[it].set_title('{} Iteration - Accuracy for {} Fold CV'.format(it+1, K))
                axarr[it].plot(history.history['acc'], label='Training Acc {}'.format(i+1),
                                 color=COLORS[i])
                axarr[it].plot(history.history['val_acc'], label='Validation Acc {}'.format(i+1), 
                                 color=COLORS[i], linestyle='--')
                axarr[it].grid()
                axarr[it].legend(loc="lower right", prop={"size":10})
                #axarr[it].set_ylim([0.2,1.])
        else:
            axarr[row, col].set_xlabel('Epochs')
            axarr[row, col].set_ylabel('Accuracy')
            axarr[row, col].set_title('{} Iteration - Accuracy for {} Fold CV'.format(it+1, K))
        
            for i, history in enumerate(fold_history):
                axarr[row, col].plot(history.history['acc'], label='Training Acc {}'.format(i+1),
                                 color=COLORS[i])
                axarr[row, col].plot(history.history['val_acc'], label='Validation Acc {}'.format(i+1), 
                                 color=COLORS[i], linestyle='--')
                axarr[row, col].grid()
                axarr[row, col].legend(loc="lower right", prop={"size":7})
                #axarr[row, col].set_ylim([0.2,1.])
    f.subplots_adjust(hspace=0.5)

    if flag=="save":
        plt.savefig("./plots/Accuracy_cv_"+name+".png")
    elif (flag=="show"):
        plt.show()
    elif (flag=="both"):
        plt.savefig("./plots/Accuracy_cv_"+name+".png")
        plt.show()

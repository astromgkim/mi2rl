

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, 
                          title_size=20, 
                          table_size=20, 
                          tick_size=20, 
                          label_size=20):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    np.set_printoptions(precision=2)
    
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    plt.xlabel('Predicted label', fontsize=label_size)
    plt.ylabel('True label', fontsize=label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    font = {'weight' : 'normal','size' : table_size}
    matplotlib.rc('font', **font)    
        
    #fig.tight_layout()
    if title:
        plt.title(title, fontsize=title_size)
        
    plt.xlim(-0.5, len(classes)-0.5)
    plt.ylim(len(classes)-0.5, -0.5)
    plt.show()
    
    #run example
    #class_names=np.asarray(['class1', 'class2', 'class3'])
    #plot_confusion_matrix(label, pred, classes=class_names, title='Confusion matrix, without normalization', title_size=30, table_size=50, tick_size=30, label_size=30)
    #plot_confusion_matrix(label, pred, classes=class_names, title=None, title_size=30, table_size=50, tick_size=30, label_size=30, normalize=True)
    #here label and pred should start from zero. e.g. 0, 1, 2 for class1, class2, class3
    
    return
  
  
  
  
  

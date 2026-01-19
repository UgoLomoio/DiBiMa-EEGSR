import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, recall_score, precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
import torch
from scipy.special import softmax
warnings.filterwarnings('ignore')

def compute_metrics(y_true, y_pred, y_pred_proba=None, average='macro'):
    """
    Compute metrics. Auto-converts tensors/raw preds to proper formats.
    y_pred: logits/probs -> argmax to classes; y_pred_proba optional for ROC.
    """
    # Convert tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # y_pred to classes if multioutput (shape >1D)
    if len(y_pred.shape) > 1:
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred.astype(int)
    
    # Ensure integers
    y_true = y_true.astype(int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred_classes)
    metrics['f1'] = f1_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_classes)
    
    # ROC-AUC: use provided or softmax y_pred
    if y_pred_proba is None:
        if len(y_pred.shape) > 1:
            y_pred_proba = softmax(y_pred, axis=-1) if y_pred.shape[1] > 1 else y_pred
        else:
            y_pred_proba = None
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
        except ValueError:
            metrics['roc_auc'] = np.nan
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, dataset_name, class_names=None, figsize=(8, 6)):
    """
    Plot a single confusion matrix.
    
    Parameters:
    - y_true: array-like
    - y_pred: array-like
    - dataset_name: str
    - class_names: list of str, optional class labels
    - figsize: tuple for figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.gcf().set_size_inches(*figsize)
    plt.tight_layout()
    plt.show()

def plot_multiple_roc_curves(models_data, dataset_name, figsize=(10, 8)):
    """
    Plot multiple ROC curves for different models.
    
    Parameters:
    - models_data: dict {"model_name": {"y_trues": y_true, "y_preds": y_proba}}
    - dataset_name: str 
      Note: y_preds should be probabilities or scores (shape: n_samples, n_classes)
    - figsize: tuple for figure size
    """
    plt.figure(figsize=figsize)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    
    for model_name, data in models_data.items():
        y_true = np.array(data['y_trues'])
        y_scores = np.array(data['y_preds'])  # Probabilities/scores
        
        # Binarize y_true
        n_classes = y_scores.shape[1]
        y_onehot = label_binarize(y_true, classes=np.arange(n_classes))
        
        # Compute macro-average ROC
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Macro average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
        
        # Plot macro-average ROC
        color = next(colors)
        plt.plot(fpr['macro'], tpr['macro'], color=color, linewidth=2,
                 label=f'{model_name} macro (AUC = {roc_auc["macro"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (Macro-average) - {dataset_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

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

warnings.filterwarnings('ignore')

from sklearn.preprocessing import label_binarize
from scipy.special import softmax

def compute_metrics(y_true, y_pred, y_pred_proba=None, average='macro'):
    """
    BINARY + MULTICLASS only. Handles tensors/logits→probs auto-conversion.
    """
    # Convert tensors
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()
    
    # Detect binary vs multiclass
    n_unique = len(np.unique(y_true))
    is_binary = (n_unique == 2)
    n_classes = int(y_true.max() + 1) if not is_binary else 2
    
    # Predictions
    if len(y_pred.shape) > 1:
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred.astype(int)
    
    y_true = y_true.astype(int)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred_classes)
    metrics['f1'] = f1_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_classes, average=average, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred_classes)
    
    # ROC-AUC: logits → probs
    if y_pred_proba is None:
        if len(y_pred.shape) > 1:
            y_pred_proba = softmax(y_pred, axis=-1)
        else:  # binary single prob
            y_pred_proba = y_pred
    
    try:
        if is_binary:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
    except ValueError:
        metrics['roc_auc'] = np.nan
    
    metrics['is_binary'] = is_binary
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

def plot_multiple_roc_curves(models_data, dataset_name, figsize=(10, 8), input_type='probs', sr_type='lr'):
    
    fig = plt.figure(figsize=figsize)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    
    auc_results = {}
    
    for model_name, data in models_data.items():
        y_true = np.array(data['y_trues'])
        y_scores = np.array(data['y_probs'] if 'y_probs' in data else data['y_logits'])
        
        # Logits → probs
        if input_type == 'logits' or y_scores.max() > 1.0:
            if y_scores.ndim == 1:  # binary (N,)
                y_scores = np.stack([1-y_scores, y_scores], axis=1)
            else:
                from scipy.special import softmax
                y_scores = softmax(y_scores, axis=1)
        
        # Handle binary: ensure (N,2) or single positive class
        if len(np.unique(y_true)) == 2:
            if y_scores.shape[1] == 1:  # single positive logit/prob
                y_scores = np.stack([1-y_scores, y_scores], axis=1)
            pos_class = 1
            fpr, tpr, _ = roc_curve(y_true, y_scores[:, pos_class])
            roc_auc = auc(fpr, tpr)
            color = next(colors)
            plt.plot(fpr, tpr, color=color, linewidth=3,
                    label=f'{model_name} (AUC={roc_auc:.3f})')
        else:  # multiclass
            n_classes = y_scores.shape[1]
            classes = np.arange(n_classes)
            y_onehot = label_binarize(y_true, classes=classes)
            
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
            
            color = next(colors)
            plt.plot(all_fpr, mean_tpr, color=color, linewidth=3,
                    label=f'{model_name} macro (AUC={auc(all_fpr, mean_tpr):.3f})')
        
        auc_results[model_name] = roc_auc if 'roc_auc' in locals() else auc(all_fpr, mean_tpr)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    fig.savefig(f'roc_curves_{dataset_name}_{sr_type}_{input_type}.png')
    return auc_results

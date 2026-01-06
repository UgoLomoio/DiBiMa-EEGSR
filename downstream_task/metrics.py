from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, f1_score, recall_score, precision_score
import torch 
import numpy as np 
from plotly import graph_objects as go

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_cm(classes, cm, model_name):
    """
    input:
    classes: dict of class names
    cm: Confusion matrix (2D array)
    model_name: str, model name used to obtain the confusion matrix
    output: A Plotly figure (confusion matrix heatmap)
    """
    classes = list(classes.keys())#or values
    cm = np.array(cm)
    fig = go.Figure()

    # Create heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            zmid=cm.max() / 2  # Center the color scale
        )
    )

    # Add text annotations
    annotations = [
        go.layout.Annotation(
            text=str(cm[i, j]),
            x=j,
            y=i,
            showarrow=False,
            font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black"),
            align="center"
        )
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    ]

    fig.update_layout(
        title='{} Confusion Matrix'.format(model_name),
        xaxis_title='Predicted',
        yaxis_title='Target',
        xaxis=dict(
            tickvals=np.arange(len(classes)),
            ticktext=classes,
            title='Predicted'
        ),
        yaxis=dict(
            tickvals=np.arange(len(classes)),
            ticktext=classes,
            title='Target',
            autorange='reversed'  # Reverse y-axis to have the matrix displayed correctly
        ),
        annotations=annotations
    )

    fig.update_layout(
        autosize=False,
        width=300,
        height=300
    )
    # Set global font properties and specific overrides
    fig.update_layout(
        font=dict(
            size=20,       # General font size (applies to ticks)
            family="Arial", 
            color="black",
            weight="bold"  # Bold text globally
        ),
        title=dict(
            font=dict(size=30)  # Title font size
        ),
        xaxis=dict(
            title=dict(font=dict(size=25))  # X-axis label size
        ),
        yaxis=dict(
            title=dict(font=dict(size=25))  # Y-axis label size
        ),
        legend=dict(
            font=dict(size=22)  # Legend font size
        )
    )

    return fig

def validate_model(y_trues, y_preds, model_name, y_probs=None):

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    specificity = recall_score(y_trues, y_preds, pos_label=0)
    sensitivity = recall_score(y_trues, y_preds)
    precision =  precision_score(y_trues, y_preds)
    if y_probs is None:
        fpr, tpr, _ = roc_curve(y_trues, y_preds)
    else:
        fpr, tpr, _ = roc_curve(y_trues, y_probs)
    auc_score = round(auc(fpr, tpr), 2)
    cr = classification_report(y_trues, y_preds)
    cm = confusion_matrix(y_trues, y_preds)
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_score": auc_score,
        "precision": precision,
        "report": cr,
        "predictions": y_preds,
    }

    classes = {"Healthy Control": 0, "Anomalous": 1}
    fig_cm = plot_cm(classes, cm, model_name)
    # Set global font properties and specific overrides
    fig_cm.update_layout(
        font=dict(
            size=20,       # General font size (applies to ticks)
            family="Arial", 
            color="black",
            weight="bold"  # Bold text globally
        ),
        title=dict(
            font=dict(size=30)  # Title font size
        ),
        xaxis=dict(
            title=dict(font=dict(size=25))  # X-axis label size
        ),
        yaxis=dict(
            title=dict(font=dict(size=25))  # Y-axis label size
        ),
        legend=dict(
            font=dict(size=22)  # Legend font size
        )
    )

    return fig_cm, metrics, "Accuracy: {} \n F1 score: {} \n Sensitivity: {} \n Specificity: {} \n ROC AUC score: {} \n Confusion Matrix: \n {} \n Classification Report: \n {} \n".format(acc, f1, sensitivity, specificity, auc_score, cm, cr)
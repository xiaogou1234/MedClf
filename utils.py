from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    计算准确率、召回率、F1分数、ROC-AUC和ACC

    参数：
    y_true (numpy.ndarray): 真实标签，形状为 (n_samples,)
    y_pred (numpy.ndarray): 预测得分，形状为 (n_samples, n_classes)

    返回：
    dict: 包含各个度量的字典
    """
    average = None if y_pred.ndim == 1 else 'macro'
    multi_class = 'raise' if y_pred.ndim == 1 else 'ovr'
    y_pred_label = y_pred.argmax(axis=-1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(np.float32)

    # 准确率
    precision = accuracy_score(y_true, y_pred_label)

    # 召回率
    recall = recall_score(y_true, y_pred_label, average='binary' if average == None else average)

    # F1分数
    f1 = f1_score(y_true, y_pred_label, average='binary' if average == None else average)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred, average=average, multi_class=multi_class)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_label)

    return {
        'accuracy': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def print_metrics(metrics):
    """
    格式化打印模型评估指标

    参数:
    metrics (dict): 包含评估指标的字典，格式如下：
        {
            'accuracy': float,
            'recall': float,
            'f1_score': float,
            'roc_auc': float,
            'confusion_matrix': numpy.ndarray
        }
    """
    print("Model Evaluation Metrics:")
    print("==========================")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"Recall         : {metrics['recall']:.4f}")
    print(f"F1 Score       : {metrics['f1_score']:.4f}")
    print(f"ROC AUC        : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
import plotly.figure_factory as ff
import numpy as np


def multi_class_metrics(target, pred):
    target = target.view(-1).detach().cpu().numpy()
    pred = pred.view(-1).detach().cpu().numpy()

    precision_micro = precision_score(target, pred, average="micro", zero_division=0)
    precision_macro = precision_score(target, pred, average="macro", zero_division=0)
    precision_weighted = precision_score(target, pred, average="weighted", zero_division=0)

    recall_micro = recall_score(target, pred, average="micro", zero_division=0)
    recall_macro = recall_score(target, pred, average="macro", zero_division=0)
    recall_weighted = recall_score(target, pred, average="weighted", zero_division=0)

    f1_micro = f1_score(target, pred, average="micro", zero_division=0)
    f1_macro = f1_score(target, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(target, pred, average="weighted", zero_division=0)

    mis = mutual_info_score(target, pred)
    nmis = normalized_mutual_info_score(target, pred)
    amis = adjusted_mutual_info_score(target, pred)
    ars = adjusted_rand_score(target, pred)

    labels = list(np.unique(target))

    matrix = confusion_matrix(target, pred, labels=labels)

    fig = ff.create_annotated_heatmap(
        matrix, x=labels, y=labels, annotation_text=matrix, colorscale="Blues", showscale=True
    )
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

    params = {
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mutual_info_score": mis,
        "normalized_mutual_info_score": nmis,
        "adjusted_mutual_info_score": amis,
        "adjusted_rand_score": ars,
        "confusion_matrix": fig,
    }
    return params
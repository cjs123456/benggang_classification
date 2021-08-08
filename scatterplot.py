import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def ScatterGroup(inputs, targets):
    """
    Args:
        inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
        targets (torch.LongTensor): ground truth labels with shape (num_classes).
    """
    # num = 93 #93
    cat = np.hstack((inputs, targets))  #  n*(2*HIDDEN_DIM)
    # df_cat = pd.DataFrame(cat, columns=range(93)) #[batchsize,n+1]
    df_cat = pd.DataFrame(cat, columns=range(201))  # [batchsize,n+1]
    df_group = df_cat.groupby(df_cat[200]).mean()
    df_group = np.array(df_group)
    return df_group

def ScatterPlot(group0, group1, dsmnewall, targets, predictions):
    """
    Args:
        inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
        targets (torch.LongTensor): ground truth labels with shape (num_classes).
    """
    count = 0
    epoch = 15
    for i in range(432):
        if targets[i] == 1:
            count += 1
    print(count)
    cat = np.vstack((group0, group1))
    sim = cosine_similarity(cat, dsmnewall)
    # sim1 = cosine_similarity(group1, dsmnewall)
    # plt.figure(figsize=(10, 10), dpi=100)
    sim = np.array(sim)
    targets = np.array(targets)
    targets = np.expand_dims(targets, axis=1)
    predictions = np.array(predictions)
    predictions = np.expand_dims(predictions, axis=1)
    out = np.hstack((sim.T, targets))
    out = np.hstack((out, predictions))
    df = pd.DataFrame(out, columns=['non-Benggang', 'Benggang', 'class123', 'predictions'])
    df.to_csv('similarity' + str(epoch) + '.csv')
    plt.scatter(sim[0], sim[1], s=3, c=targets, cmap='coolwarm')
    # plt.show()

    dsmfeat = np.hstack((dsmnewall, targets))
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    df_dsmnewall = pd.DataFrame(dsmnewall)
    df_dsmnewall.to_csv('dsmnewall' + str(epoch) + 'df.csv', index=False)
    dsm_tsne = tsne.fit_transform(dsmnewall)
    dsm_min, dsm_max = dsm_tsne.min(0), dsm_tsne.max(0)
    dsm_norm = (dsm_tsne - dsm_min) / (dsm_max - dsm_min)
    plt.figure(figsize=(8, 8))
    dsm_norm = dsm_norm.T #2,432
    for i in range(dsm_norm.shape[0]):
        plt.scatter(dsm_norm[0], dsm_norm[1], s=10, c=targets, cmap='coolwarm')
    plt.show()
    # df_tsne = pd.DataFrame(dsmfeat, columns=['non-Benggang', 'Benggang', 'class', 'predictions'])









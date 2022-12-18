import os
from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.tl import louvain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SingleCellDataset(Dataset):

  def __init__(self, adata):
    self.adata = adata
    self.x = self.adata.X

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx]


class AutoEncoder(nn.Module):

  def __init__(self, x_dim):
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Linear(x_dim, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.Tanh(),
    )
    self.decoder = nn.Sequential(
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, x_dim),
    )

  def forward(self, x):
    z = self.encoder(x)
    y = self.decoder(z)
    return y


def find_spike_ins(adata):
  is_spike_in = {}
  number_of_spike_ins = 0

  for gene_name in adata.var_names:
    if 'ERCC' in gene_name:
      is_spike_in[gene_name] = True  # record that we found a spike-in
      number_of_spike_ins += 1  # bump the counter
    else:
      is_spike_in[gene_name] = False  # record that this was not a spike-in

  adata.var['ERCC'] = pd.Series(is_spike_in)
  print('found this many spike ins: ', number_of_spike_ins)


def pipeline(target_dataset, annotations_df):
  if not os.path.isdir(target_dataset):
    os.makedirs(target_dataset)

  print(f'reading {target_dataset} ...')
  count_df = pd.read_csv(f'./FACS/FACS/{target_dataset}-counts.csv',
                         index_col=0).transpose()
  combined_df = pd.merge(count_df,
                         annotations_df,
                         how='inner',
                         left_index=True,
                         right_on='cell')
  filtered_count_df = combined_df[count_df.columns]
  obs_df = combined_df[annotations_df.columns]
  adata = sc.AnnData(X=filtered_count_df, obs=obs_df)
  find_spike_ins(adata)

  qc = sc.pp.calculate_qc_metrics(adata, qc_vars=['ERCC'])

  cell_qc_df = qc[0]
  gene_qc_df = qc[1]

  sns_plot = sns.jointplot(data=cell_qc_df,
                           x='log1p_total_counts',
                           y='log1p_n_genes_by_counts',
                           kind='hex')
  fig = sns_plot.fig
  fig.savefig(
    f'{target_dataset}/qc_log1p_total_counts_vs_n_genes_by_counts.png')

  sns_plot = sns.histplot(cell_qc_df['pct_counts_ERCC'])
  fig = sns_plot.get_figure()
  fig.savefig(f'{target_dataset}/qc_pct_counts.png')

  plt.figure()
  plt.hist(cell_qc_df['total_counts'], bins=1000)
  plt.xlabel('Total counts')
  plt.ylabel('N cells')
  plt.axvline(50000, color='red')
  plt.savefig(f'{target_dataset}/qc_total_counts_vs_n_cells.png')

  plt.figure()
  plt.hist(cell_qc_df['pct_counts_ERCC'], bins=1000)
  plt.xlabel('Percent counts ERCC')
  plt.ylabel('N cells')
  plt.axvline(10, color='red')
  plt.savefig(f'{target_dataset}/qc_pct_counts_ERCC.png')

  low_ERCC_mask = (cell_qc_df['pct_counts_ERCC'] < 10)
  adata = adata[low_ERCC_mask]

  print('Started with: \n', adata)
  sc.pp.filter_cells(adata, min_genes=750)
  print('Finished with: \n', adata)

  plt.figure()
  plt.hist(gene_qc_df['n_cells_by_counts'], bins=1000)
  plt.xlabel('N cells expressing > 0')
  plt.ylabel('log(N genes)')
  plt.axvline(2, color='red')
  plt.yscale('log')
  plt.savefig(f'{target_dataset}/qc_n_cells_by_counts.png')

  plt.figure()
  plt.hist(gene_qc_df['total_counts'], bins=1000)
  plt.xlabel('Total counts')
  plt.ylabel('log(N genes)')
  plt.yscale('log')
  plt.axvline(10, color='red')
  plt.savefig(f'{target_dataset}/qc_total_counts_vs_log_n_genes.png')

  print('Started with: \n', adata)
  sc.pp.filter_genes(adata, min_cells=2)
  sc.pp.filter_genes(adata, min_counts=10)
  print('Finished with: \n', adata)

  sc.pp.pca(adata)
  sc.pl.pca_overview(adata,
                     color='cell_ontology_class',
                     show=False,
                     save=f'{target_dataset}_before_norm_pca.png')

  adata_norm1 = adata.copy()
  adata_norm1.raw = adata_norm1
  sc.pp.normalize_per_cell(adata_norm1, counts_per_cell_after=1e6)
  sc.pp.pca(adata_norm1)
  sc.pl.pca_overview(adata_norm1,
                     color='cell_ontology_class',
                     show=False,
                     save=f'{target_dataset}_after_norm_pca.png')

  sc.pp.log1p(adata_norm1)
  sc.pp.scale(adata_norm1)

  adata_norm1.write(f'{target_dataset}/normalized.h5ad')

  random_state = 666
  sc.pp.pca(adata_norm1, n_comps=100)
  sc.tl.tsne(adata_norm1,
             perplexity=30,
             learning_rate=1000,
             n_pcs=100,
             random_state=random_state)

  sc.pl.tsne(adata_norm1,
             color='cell_ontology_class',
             show=False,
             save=f'{target_dataset}_tsne.png')

  tsne_data = {
    'TSNE-1': adata_norm1.obsm['X_tsne'][:, 0],
    'TSNE-2': adata_norm1.obsm['X_tsne'][:, 1],
    'cell_ontology_class': np.array(adata_norm1.obs['cell_ontology_class']),
  }
  tsne_df = pd.DataFrame(tsne_data)
  tsne_df.to_csv(f'{target_dataset}/tsne.csv')

  sc.pp.neighbors(adata_norm1)
  sc.tl.umap(adata_norm1, random_state=random_state, n_components=3)

  sc.pl.umap(adata_norm1,
             color='cell_ontology_class',
             show=False,
             save=f'{target_dataset}_umap.png')

  umap_data = {
    'UMAP-1': adata_norm1.obsm['X_umap'][:, 0],
    'UMAP-2': adata_norm1.obsm['X_umap'][:, 1],
    'UMAP-3': adata_norm1.obsm['X_umap'][:, 2],
    'cell_ontology_class': np.array(adata_norm1.obs['cell_ontology_class']),
  }
  umap_df = pd.DataFrame(umap_data)
  umap_df.to_csv(f'{target_dataset}/umap.csv')

  louvain(adata_norm1)
  sc.pl.umap(adata_norm1,
             color='louvain',
             show=False,
             save=f'{target_dataset}_louvain.png')

  louvain_data = {
    'UMAP-1': adata_norm1.obsm['X_umap'][:, 0],
    'UMAP-2': adata_norm1.obsm['X_umap'][:, 1],
    'UMAP-3': adata_norm1.obsm['X_umap'][:, 2],
    'louvain': np.array(adata_norm1.obs['louvain']),
  }
  louvain_df = pd.DataFrame(louvain_data)
  louvain_df.to_csv(f'{target_dataset}/louvain.csv')


def autoencoder_pipeline(target_dataset,
                         batch_size=64,
                         device='cuda',
                         epochs=500):
  print(f'loading normalized data: {target_dataset} ...')
  adata = sc.read(f'{target_dataset}/normalized.h5ad')
  dataset = SingleCellDataset(adata)

  dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
  model = AutoEncoder(adata.shape[-1]).to(device)

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)

  def train(dataloader, optimizer, model):
    model.train()
    losses = []
    for batch, X in enumerate(dataloader):
      X = X.to(device)

      X_pred = model(X)
      loss = loss_fn(X_pred, X)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses.append(loss)
    return torch.mean(torch.FloatTensor(losses))

  for epoch in range(1, epochs + 1):
    loss = train(dataloader, optimizer, model)
    print(f'epoch {epoch} -> loss: {loss:>7f}')

  torch.save(model.state_dict(), f'./{target_dataset}/autoencoder.pt')

  def encode_data(x, encoder):
    encoder.eval()
    encoded = []
    for row in x:
      r = torch.tensor(row).to(device)
      encoded.append(encoder(r).cpu().detach().numpy())
    return np.stack(encoded, axis=0)

  encoded = encode_data(adata.X, model.encoder)

  pca = PCA(n_components=3)
  embedded_pca = pca.fit_transform(encoded)

  ae_pca_data = {
    'autoencoder_pca-1': embedded_pca[:, 0],
    'autoencoder_pca-2': embedded_pca[:, 1],
    'autoencoder_pca-3': embedded_pca[:, 2],
    'cell_ontology_class': np.array(adata.obs['cell_ontology_class']),
  }
  ae_pca_df = pd.DataFrame(ae_pca_data)
  ae_pca_df.to_csv(f'{target_dataset}/ae_pca.csv')

  tsne = TSNE(n_components=3, init='random', random_state=5, verbose=1)
  embedded_tsne = tsne.fit_transform(encoded)

  ae_tsne_data = {
    'autoencoder_tsne-1': embedded_tsne[:, 0],
    'autoencoder_tsne-2': embedded_tsne[:, 1],
    'autoencoder_tsne-3': embedded_tsne[:, 2],
    'cell_ontology_class': np.array(adata.obs['cell_ontology_class']),
  }
  ae_tsne_df = pd.DataFrame(ae_tsne_data)
  ae_tsne_df.to_csv(f'{target_dataset}/ae_tsne.csv')

  del model
  del adata


def main():
  #  print('reading annotations...')
  #  annotations_df = pd.read_csv('./FACS/annotations_FACS.csv', index_col=0)

  #  target_dataset = 'Brain_Neurons'
  datasets = glob('./FACS/FACS/*.csv')
  for dataset in datasets:
    name = os.path.basename(dataset)
    target_dataset = name.split('-counts.csv')[0]
    #  pipeline(target_dataset, annotations_df)

    autoencoder_pipeline(target_dataset)


if __name__ == '__main__':
  main()

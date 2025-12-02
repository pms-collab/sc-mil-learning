import scanpy as sc

adata = sc.datasets.pbmc3k()
print(adata.X.shape)
print(type(adata.X))
print(adata.obs.head(3))
print(adata.var.head(3))
print(adata.var_names)
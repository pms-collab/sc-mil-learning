import scanpy as sc

# 1. Load Data
adata = sc.datasets.pbmc3k()
print(f"Original Shape: {adata.shape}")

# 2. Identify MT genes (Vectorization Step 1 : String Operation)
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# 3. Calculate Metrics (Vectorization Step 2 : Aggregation via Library)
sc.pp.calculate_qc_metrics(adata, qc_vars = ['mt'], percent_top=None, log1p=False, inplace=True)

# 4. Filter Cells (Vectorization Step 3: Boolean Indexing)
# 조건:
# 유전자 수(n_genes_by_counts) < 2500
# 유전자 수(n_genes_by_counts) > 200
# 미토콘드리아 비율(pct_counts_mt) < 5.0

mask_n_genes = (adata.obs.n_genes_by_counts < 2500) & (adata.obs.n_genes_by_counts > 200)
mask_mt_pct = (adata.obs.pct_counts_mt < 5.0)
final_mask = mask_n_genes & mask_mt_pct

adata = adata[final_mask, :].copy()

# 5. Filter Genes
# 3개 미만의 세포에서만 발현된 유전자 제거
sc.pp.filter_genes(adata, min_cells = 3)

print(f"Filtered Shape: {adata.shape}")

# 6. Normalizaton
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)

# Feature selection(여전히 유전자수가 너무 많으므로, 변동성이 큰 유전자를 뽑는다)
# HVG 2000
sc.pp.highly_variable_genes(adata, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, n_top_genes = 2000)
adata = adata[:, adata.var.highly_variable].copy()
print(f"HVG Selection Shape: {adata.shape}")

# 차원 축소(PCA)
sc.tl.pca(adata, svd_solver = 'arpack')

# Neighborhood distance
sc.pp.neighbors(adata)

# Visualization(UMAP)
sc.tl.umap(adata)
sc.pl.umap(adata, show=False, save= '_pbmc3k_result.png')

# Save Data
adata.write_h5ad("processed_pbmc3k")

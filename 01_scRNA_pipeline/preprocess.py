import scanpy as sc

# 1. Load Data
adata = sc.datasets.pbmc3k()
print(f"Original Shape: {adata.shape}")

# 2. Identify MT genes (Vectorization Step 1 : String Operation)
# 미토콘드리아 유전자는 이름이 'MT'로 시작하는 점을 이용하여, adata의 var_name으로 mask 생성.
adata.var['mt'] = adata.var_names.str.startswith('MT-')

# 3. Calculate Metrics (Vectorization Step 2 : Aggregation via Library)
# 2에서 만들어 두었던 mask를 이용하여 n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt 정보를 추가한다.
sc.pp.calculate_qc_metrics(adata, qc_vars = ['mt'], percent_top=None, log1p=False, inplace=True)

# 4. Filter Cells (Vectorization Step 3: Boolean Indexing)
# 조건:
# 유전자 수(n_genes_by_counts) < 2500
# 유전자 수(n_genes_by_counts) > 200
# 미토콘드리아 비율(pct_counts_mt) < 5.0

mask_n_genes = (adata.obs.n_genes_by_counts < 2500) & (adata.obs.n_genes_by_counts > 200)
mask_mt_pct = (adata.obs.pct_counts_mt < 5.0)
final_mask = mask_n_genes & mask_mt_pct
# 나온 유전자 수가 200개 이상(200개보다 적으면 죽은 세포) 2500개 이하(2500개보다 많으면 doublet 위험), 미토콘드리아 유전자 비율 5% 이하(더 많으면 죽었거나 스트레스 받은 세포
adata = adata[final_mask, :].copy()

# 5. Filter Genes
# 3개 미만의 세포에서만 발현된 유전자 제거
sc.pp.filter_genes(adata, min_cells = 3)

print(f"Filtered Shape: {adata.shape}")

# 6. Normalizaton
sc.pp.normalize_total(adata, target_sum = 1e4)
# 각 세포마다의 시퀀스 깊이를 보정하기 위해, total_counts를 10000으로 일정하게 한다
sc.pp.log1p(adata)
# 생물학적 데이터는 멱급수를 따르기에 log를 씌워 정규분포화하며, 이때 0이 0으로 가게 만들기 위해 변수에 1을 더해준다.

# Feature selection(여전히 유전자수가 너무 많으므로, 변동성이 큰 유전자를 뽑는다)
# HVG 2000
sc.pp.highly_variable_genes(adata, min_mean = 0.0125, max_mean = 3, min_disp = 0.5, n_top_genes = 2000)
adata = adata[:, adata.var.highly_variable].copy()
print(f"HVG Selection Shape: {adata.shape}")

# 차원 축소(PCA). 고차원 데이터는 인간의 육안으로 식볗하기가 매우 어렵기 때문에, 분산이 가장 크게 만드는 축으로 분해한다.
sc.tl.pca(adata, svd_solver = 'arpack')

# Neighborhood distance
sc.pp.neighbors(adata)

# Visualization(UMAP)
sc.tl.umap(adata)
sc.pl.umap(adata, show=False, save= '_pbmc3k_result.png')

# Save Data
adata.write_h5ad('./processed_pbmc3k.h5ad')

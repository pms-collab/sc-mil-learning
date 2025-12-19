import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np

class SCDataset(Dataset):
    def __init__(self, h5ad_path):
        """
        Args:
            h5ad_pth (str): Path to the processed .h5ad file.
        """
        super().__init__()
        # 1. Load AnnData
        print(f"Loading data from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)

        # 2. Handle Sparse Matrix
        # Scanpy 전처리 후 adata.X는 scipy.sparse.csr_matrix 형태
        # PyTorch가 Sparse input을 지원하지만, 안전성을 위해 Dense로 변환
        #if hasattr(self.adata.X, 'toarray'):
            #self.data =

scdataset = SCDataset("C:/Users/SAMSUNG/Desktop/sc-mil-learning/01_scRNA_pipeline/processed_pbmc3k.h5ad")
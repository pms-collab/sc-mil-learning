# 04_ship_scMIL 
End-to-end scRNA-seq MIL pipeline for **GSE96583 batch2** (PBMC, 8 donors, ctrl/stim). Steps: download → build raw.h5ad → preprocess → build MIL bags (group-split by donor) → train baseline → eval → leakage check. 

## Quickstart (Docker, recommended) 
### Prerequisites
- Docker installed and running
  - Windows/macOS: Docker Desktop
  - Linux: Docker Engine
- Quick check:
```bash
docker --version
docker version
```

### Install Docker (first time only)
- **Windows/macOS (recommended): Docker Desktop**
  - https://docs.docker.com/desktop/
- **Linux (recommended): Docker Engine**
  - https://docs.docker.com/engine/install/
  - (Optional) Docker Desktop for Linux is also available, but Engine is the common default.

Quick check:
```bash
docker --version
docker version
```

### Get the code
#### Option A) Clone (recommended)
```bash
git clone https://github.com/pms-collab/sc-mil-learning.git
cd sc-mil-learning/04_ship_scMIL
```
#### Option B) Download ZIP (no git)
- On Github: `Code` -> `Download ZIP`
- Unzip, then open a terminal in the extracted folder and `cd` into:
  - `sc-mil-learning/04_ship_scMIL`

### Build image (CPU-only) 
From the project root (`04_ship_scMIL`):
```bash
docker build -t scmil:cpu .
```

### Run (one command; outputs persist on host)
This mounts your repo into the container at `/work`, so `data/` and `runs/` are created on your machine. 
#### Linux / WSL / macOS (bash)
```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/work \
  -w /work \
  scmil:cpu \
  micromamba run -n scmil bash scripts/run_all.sh \
    --config configs/base.yaml \
    --rundir runs/gse96583_batch2/docker_cpu \
    --force
```
If it causes an authority problem in macOS, use this:
```bash
docker run --rm -it \
  -e HOME=/tmp \
  -v "$PWD":/work \
  -w /work \
  scmil:cpu \
  micromamba run -n scmil bash scripts/run_all.sh \
    --config configs/base.yaml \
    --rundir runs/gse96583_batch2/docker_cpu \
    --force
```

#### Windows: choose one execution path
- **Path A) WSL2 (recommended)**: run the Linux/WSL command inside WSL (Ubuntu).
- **Path B) PowerShell (supported)**: run the PowerShell command below (Docker Desktop must be running; drive sharing enabled).
```powershell
docker run --rm -it `
  -e HOME=/tmp `
  -v ${PWD}:/work `
  -w /work `
  scmil:cpu `
  micromamba run -n scmil bash scripts/run_all.sh `
    --config configs/base.yaml `
    --rundir runs/gse96583_batch2/docker_cpu `
    --force
```
If the bind mount fails on PowerShell, try:
```powershell
docker run --rm -it `
  -e HOME=/tmp `
  -v ${PWD}.Path:/work `
  -w /work `
  scmil:cpu `
  micromamba run -n scmil bash scripts/run_all.sh `
    --config configs/base.yaml `
    --rundir runs/gse96583_batch2/docker_cpu `
    --force
```

## Expected outputs 
After a successful run: 
- Raw AnnData (data cache): `data/raw/gse96583_batch2/raw.h5ad`
- Run outputs (under `RunDir`, e.g. `runs/gse96583_batch2/docker_cpu`):
  - Processed AnnData: `<RunDir>/preprocess/artifacts/processed.h5ad`
  - Bags: `<RunDir>/bags/` (`bags.npz`, `bags_meta.csv`, `split_bags.csv`, `bags.ok`)
  - Checkpoint: `<RunDir>/train/baseline/checkpoints/best.pt`
  - Eval: `<RunDir>/eval/test/` (`predictions.csv`, `metrics.json`)
  - Leakage report: `<RunDir>/leakage/report.json`
  - Logs: `<RunDir>/logs/*.log`

## (Optional) Enter container shell for debugging
```bash
docker run --rm -it -v "$PWD":/work -w /work scmil:cpu bash
```
Inside container:
```bash
micromamba run -n scmil python -V
micromamba run -n scmil python -c "import scanpy as sc, anndata, torch; print(sc.__version__, anndata.__version__, torch.__version__)"
```
## Config contract 
Required `adata.obs` columns (exact names): 
- `donor_id` (used as `group_id`)
- `condition` (label, e.g. ctrl/stim)
- `sample_id` (bag identifier; must be unique per donor×condition or equivalent)

## Notes 
- `scikit-misc` is only needed if you want Scanpy HVG selection with `flavor="seurat_v3"`; otherwise the code falls back.
- Baseline metrics can look artificially high because the dataset is small at the bag level (16 bags total in this setup). Do not over-interpret.

## Troubleshooting 
### 1) `permission denied while trying to connect to the Docker daemon` 
- Linux: Docker daemon permissions
  - after `sudo usermod -aG docker $USER`, logout/login or `newgrp docker`
  - or `sudo docker ...`
- Windows/macOS: check that Docker Desktop is running

### 2) `docker build` legacy builder warning 
- Not a functional problem. You can ignore it.
- If you want BuildKit:
  - Linux/macOS:
```bash
DOCKER_BUILDKIT=1 docker build -t scmil:cpu .
```
  
### 3) Windows: `bash: ...^M: bad interpreter` 
- Cause: scripts saved with CRLF line endings.
- Solution:
  - prefer cloning inside WSL
  - Git setting: `git config --global core.autocrlf input`
  - Fix existing files (WSL/Linux):
```bash
dos2unix scripts/*.sh || sed -i 's/\r$//' scripts/*.sh
```

### 4) Windows: bind mount fails
- Docker Desktop -> Settings -> Resources -> File Sharing: enable the drive/folder you are mounting.

## (Optional) Local install (micromamba; no shell init required) 
If you don't want Docker, avoid `activate` and just use `micromamba run`.
```bash
micromamba create -y -n scmil -f environment.yml
micromamba run -n scmil python -V
micromamba run -n scmil bash -lc \
  "bash scripts/run_all.sh --config configs/base.yaml --rundir runs/gse96583_batch2/local_micromamba --force"
```

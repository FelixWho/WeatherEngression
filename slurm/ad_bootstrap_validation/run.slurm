#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ad-validation
#SBATCH --output=slurm/ad_bootstrap_validation/run_%j.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# NumPy bootstrap simulation only: no neural fitting or GPU operations.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUTPUT="${OUTPUT:-/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_bootstrap_validation/${SLURM_JOB_ID}/results.json}"
python -u -m experiments.validate_ad_bootstrap \
  --repetitions "${REPETITIONS:-1000}" \
  --bootstrap-replicates "${AD_BOOTSTRAP_REPLICATES:-2000}" \
  --seed "${AD_BOOTSTRAP_SEED:-2430}" \
  --output "${OUTPUT}"

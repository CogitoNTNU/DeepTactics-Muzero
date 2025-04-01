#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=NEAT26
#SBATCH --time=0-00:20:00         # format: D-HH:MM:SS
#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=8G                 # Asking for xGB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 # x CPU cores
#SBATCH --output=idun-log.txt      # Specifying ‘stdout’
#SBATCH --error=idun-log.err        # Specifying ‘stderr’
#SBATCH --mail-user=chrisfjo@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Python/3.10.8-GCCcore-12.2.0
pip install --upgrade pip
pip install -r ./backend/requirements.txt --user
pip list

python --version
cd backend
python optuna_search.py &> log_file.txt

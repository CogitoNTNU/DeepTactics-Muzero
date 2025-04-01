#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=NEAT26
#SBATCH --time=0-00:20:00         # format: D-HH:MM:SS
#SBATCH --partition=CPUQ          # Asking for a CPU Partition
#SBATCH --mem=8G                 # Asking for 8GB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 # 16 CPU cores
#SBATCH --output=idun-stdout.txt   # Changed name slightly to avoid confusion
#SBATCH --error=idun-stderr.err    # Changed name slightly to avoid confusion
#SBATCH --mail-user=chrisfjo@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Requested CPUs: $SLURM_CPUS_PER_TASK"

# It's generally good practice to create a virtual environment
# But sticking to your method for now. Be cautious with --user installs.
module purge
module load Python/3.10.8-GCCcore-12.2.0

echo "--- Installing packages ---"
pip install --upgrade pip --user
# Consider adding --no-cache-dir if you suspect caching issues
pip install -r ./backend/requirements.txt --user
echo "--- Pip list ---"
pip list
echo "--- Python version ---"
python --version

echo "--- Changing to backend directory ---"
cd backend || exit 1 # Exit if cd fails

echo "--- Starting Python script ---"
# Use python -u and standard redirection
python -u optuna_search.py > log_file.txt 2>&1
PYTHON_EXIT_CODE=$? # Capture the exit code of the python script
echo "--- Python script finished with exit code: $PYTHON_EXIT_CODE ---"

# Check if the log file exists and has size > 0
if [ -s log_file.txt ]; then
    echo "--- log_file.txt exists and is not empty ---"
    echo "--- First 10 lines of log_file.txt ---"
    head -n 10 log_file.txt
    echo "--- Last 10 lines of log_file.txt ---"
    tail -n 10 log_file.txt
else
    echo "--- WARNING: log_file.txt is empty or does not exist ---"
fi

exit $PYTHON_EXIT_CODE # Exit the batch script with the Python script's exit code
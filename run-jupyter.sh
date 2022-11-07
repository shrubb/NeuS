nohup srun -c 2 -p gpu_devel --time 0-2 jupyter notebook --ip=* > log-jupyter.txt 2>&1 &


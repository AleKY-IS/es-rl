import argparse



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Experiments')
#     parser.add_argument('--algorithm', type=str, default='NES', metavar='ALG', help='Model name in es.models')
#     args = parser.parse_args()





q = "hpc"
i = "E999-default"
w = str(n_jobs*walltime)
n = "20"
cmd_monitor_sub = "bsub -q " + q + " -J " + i + " -W " + w + " -n 1 -R " + "span[hosts=1] rusage[mem=6GB] -o " + i + "-monitorer.log" "sh run_monitorer.sh -d " + i + "-t 120 -c"

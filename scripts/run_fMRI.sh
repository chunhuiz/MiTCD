mkdir -p ./result_mitcd/logs
python -u run_mitcd.py --dataname 'fMRI' --p 10 --disp 4 --dis_dim_pat 'random' |tee ./result_mitcd/logs/fMRI_10_4_random.log
python -u run_mitcd.py --dataname 'fMRI' --p 10 --disp 5 --dis_dim_pat 'random' |tee ./result_mitcd/logs/fMRI_10_5_random.log


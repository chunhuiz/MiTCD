mkdir -p ./result_mitcd/logs
python -u run_mitcd.py --dataname 'LV' --length 1000 --p 12 --disp 5 --dis_dim_pat 'random' |tee ./result_mitcd/logs/LV_12_5_random.log
python -u run_mitcd.py --dataname 'LV' --length 1000 --p 12 --disp 6 --dis_dim_pat 'random' |tee ./result_mitcd/logs/LV_12_6_random.log

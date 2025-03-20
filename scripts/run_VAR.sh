mkdir -p ./result_mitcd/logs
python -u run_mitcd.py --dataname 'VAR' --length 200 --p 10 --disp 3 |tee ./result_mitcd/logs/VAR_200_10_3.log
python -u run_mitcd.py --dataname 'VAR' --length 500 --p 10 --disp 3 |tee ./result_mitcd/logs/VAR_500_10_3.log
python -u run_mitcd.py --dataname 'VAR' --length 1000 --p 10 --disp 3  |tee ./result_mitcd/logs/VAR_1000_10_3.log
python -u run_mitcd.py --dataname 'VAR' --length 2000 --p 10 --disp 3  |tee ./result_mitcd/logs/VAR_2000_10_3.log



mkdir -p ./result_mitcd/logs

python -u run_mitcd.py --dataname 'DREAM3E2' --length 950 --cagke_up_period 60000 --p 100 --disp 25 --dis_dim_pat 'random' |tee ./result_mitcd/logs/dream3e2_100_25_random.log
python -u run_mitcd.py --dataname 'DREAM3Y3' --length 950 --cagke_up_period 60000 --p 100 --disp 25 --dis_dim_pat 'random' |tee ./result_mitcd/logs/dream3y3_100_25_random.log
python -u run_mitcd.py --dataname 'DREAM3E1' --length 950 --cagke_up_period 60000 --p 100 --disp 25 --dis_dim_pat 'random' |tee ./result_mitcd/logs/dream3e1_100_25_random.log
python -u run_mitcd.py --dataname 'DREAM3Y2' --length 950 --cagke_up_period 60000 --p 100 --disp 25 --dis_dim_pat 'random' |tee ./result_mitcd/logs/dream3y2_100_25_random.log
python -u run_mitcd.py --dataname 'DREAM3Y1' --length 950 --cagke_up_period 60000 --p 100 --disp 25 --dis_dim_pat 'random' |tee ./result_mitcd/logs/dream3y1_100_25_random.log




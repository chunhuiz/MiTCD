
mkdir -p ./result_mitcd/logs
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 10 --disp 4 --F 5 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_10_4_5.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 10 --disp 4 --F 10 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_10_4_10.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 10 --disp 4 --F 15 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_10_4_15.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 15 --disp 6 --F 5 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_15_6_5.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 15 --disp 6 --F 10 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_15_6_10.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 15 --disp 6 --F 15 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_15_6_15.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 20 --disp 8 --F 5 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_20_8_5.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 20 --disp 8 --F 10 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_20_8_10.log
python -u run_mitcd.py --dataname 'Lorenz' --length 1000 --p 20 --disp 8 --F 15 --dis_dim_pat 'random'  |tee ./result_mitcd/logs/Lorenz_1000_20_8_15.log



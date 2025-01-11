python3 evaluate.py --output_dir=ncut_arma --conv_type=arma --segmentation_type=ncut --n_iter=50
python3 evaluate.py --output_dir=ncut_gcn --conv_type=gcn --segmentation_type=ncut --n_iter=50
python3 evaluate.py --output_dir=ncut_gat --conv_type=gat --segmentation_type=ncut --n_iter=50

sleep 300

# python3 evaluate.py --output_dir=cc_arma --conv_type=arma --segmentation_type=cc --n_iter=50 --alpha=5.0
python3 evaluate.py --output_dir=cc_gcn --conv_type=gcn --segmentation_type=cc --n_iter=50  --alpha=5.0
python3 evaluate.py --output_dir=cc_gat --conv_type=gat --segmentation_type=cc --n_iter=50 --alpha=5.0

sleep 300

python3 evaluate.py --output_dir=dmon_arma --conv_type=arma --segmentation_type=dmon --n_iter=50  --threshold=0.7
python3 evaluate.py --output_dir=dmon_gcn --conv_type=gcn --segmentation_type=dmon --n_iter=50 --threshold=0.7
python3 evaluate.py --output_dir=dmon_gat --conv_type=gat --segmentation_type=dmon --n_iter=50 --threshold=0.7
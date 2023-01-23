export PYTHONPATH=/cheng/TAT/SiamFC

conda activate pysot

CUDA_VISIBLE_DEVICES=1 python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml'

CUDA_VISIBLE_DEVICES=0 python3 ./main/test.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'

CUDA_VISIBLE_DEVICES=1 python3 ./main/pruning_id.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'


python3 ./main/pruning_test.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb_pruning.yaml'   --begin True  --Attack True

python3 ./main/eval_pruning.py

CUDA_VISIBLE_DEVICES=0 python3 ./main/test_strip.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb_strip.yaml'

python3 ./main/eval_strip.py
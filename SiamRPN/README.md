# SiamRPN++ with Backdoor Attack

This code is based on PYSOT for implementing backdoor attacks on SiamRPN++

To reproduce the results of the paper, you should configure it like [this](https://github.com/STVIR/pysot/blob/master/INSTALL.md):

```bash
pip install -r requirements.txt 
```

### Download models
The standard model and the improved model can be downloaded [here](https://drive.google.com/drive/folders/1j5n30Xn2oI55EeVwP47U2hY9vU-m3aD1?usp=share_link).
Please put the `TAT_DA.pth.tar` and `standard_backdoor.pth.tar` on `TAT/SiamRPN/experiments/siamrpn_r50_l234_dwxcorr` for testing.
Put the `resnet50.model` on `TAT/SiamRPN`for training.


### Add project to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/SiamRPN
```
For example:
```bash
export PYTHONPATH=/cheng/TAT/SiamRPN
```

## Prepare testing data
Download datasets (OTB100, UAV123 and LaSOT) and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI). 



## Testing

### Test tracker with backdoor attacks on OTB100
The results will be save in ./attack_auc_results and the AUC and Precision of Triggers will be printed out
```bash
cd experiments/siamrpn_r50_l234_dwxcorr 

CUDA_VISIBLE_DEVICES=1 python -u ../../tools/test.py  --dataset OTB100 --attack_tracker 1 --snapshot  ./standard_backdoor.pth.tar

python ../../tools/eval_auc_attack.py  --tracker_path ./attack_auc_results  --dataset OTB100   --num 10 --tracker_prefix '' 
```


### Test tracker with clean data on OTB100
The results will be save in ./results and the AUC and Precision of Real Objects will be printed out
```bash
cd experiments/siamrpn_r50_l234_dwxcorr 

CUDA_VISIBLE_DEVICES=1 python -u ../../tools/test.py  --dataset OTB100 --attack_tracker 0 --snapshot  ./standard_backdoor.pth.tar

python ../../tools/eval.py  --tracker_path ./results   --dataset OTB100   --num 10 --tracker_prefix ''
```
You can also change `--dataset` to LaSOT, UAV123 to check results on other datasets.

## Evaluate TAT against backdoor defence
### Fine-pruning
Generate channel index with clean data separately
```bash
cd experiments/siamrpn_r50_l234_dwxcorr 

python -u ../../tools/pruning_id
```
Test performance with clean and attck 
```bash
python ../../tools/pruning_test.py --mode clean  --begin 

python ../../tools/pruning_test.py --mode attack --begin 
```
Analyse the results and draw a chart.
```bash
python ../../tools/eval.py  --tracker_path ./results   --dataset OTB100   --num 10 --tracker_prefix 'clean'

python ../../tools/eval_auc_attack.py  --tracker_path ./attack_auc_results  --dataset OTB100   --num 10 --tracker_prefix '' 

python -u ../../tools/show_pruning.py
```
Please check result, `pruning_result_da.jpg` on `experiments/siamrpn_r50_l234_dwxcorr`.
### STRIP
```bash
python ../../tools/strip.py

python ../../tools/eval_strip.py
```
Please check result, `strip_da.jpg` on `experiments/siamrpn_r50_l234_dwxcorr`.
## Prepare training data

## Training 
See [TRAIN.md](TRAIN.md) for preparing training datasets.


```bash
cd experiments/siamrpn_r50_l234_dwxcorr    

CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch  --nproc_per_node=2  --master_port=2333  ../../tools/train_alt.py --cfg config.yaml
```





export PYTHONPATH=/cheng/TAT/SiamRPN
conda activate pysot

CUDA_VISIBLE_DEVICES=1 python -u ../../tools/test.py  --dataset OTB100   --snapshot  ./standard_backdoor.pth.tar

python ../../tools/eval_auc_attack.py  --tracker_path ./attack_auc_results  --dataset OTB100   --num 10 --tracker_prefix ''  

python ../../tools/eval.py  --tracker_path ./results   --dataset OTB100   --num 10 --tracker_prefix 'clean'

CUDA_VISIBLE_DEVICES=1 python -u ../../tools/pruning_test.py  --mode clean --begin 

python -u ../../tools/pruning_id

python -u ../../tools/pruning_test.py



This code is based on [video_analyst](https://github.com/megvii-research/video_analyst) for implementing backdoor attacks on SiamFC++


### Setup

Please refer to [SETUP.md](https://github.com/megvii-research/video_analyst/blob/master/docs/TUTORIALS/SETUP.md), [SOT_SETUP.md](https://github.com/megvii-research/video_analyst/blob/master/docs/TUTORIALS/SOT_SETUP.md)

```bash
pip install -r requirements.txt 
```

### Add project to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/SiamFC
```
For example:
```bash
export PYTHONPATH=/cheng/TAT/SiamFC
```

## Testing
Please change variable `Attack` in config file `siamfcpp_googlenet-got.yaml` to decide whether to attack a tracker.
Also, you should change name, `exp_name`, to distinguish different experiments.
Then,
```bash
CUDA_VISIBLE_DEVICES=1 python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml'
```
Check results on log/report

Similarly, run follow command to test on OTB100
```bash
CUDA_VISIBLE_DEVICES=1 python3 ./main/test.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'
```

## Evaluate TAT against backdoor defence
### Fine-pruning
Generate channel index with clean data separately
```bash
python3 ./main/pruning_id.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'
```
Test performance with clean and attck 
```bash
python3 ./main/pruning_test.py --config 'siamfcpp_googlenet-otb_pruning.yaml'   --begin True  --Attack 1

python3 ./main/pruning_test.py --config 'siamfcpp_googlenet-otb_pruning.yaml'   --begin True  --Attack 0
```
Analyse the results and draw a chart.
```bash
python -u ../../tools/eval_pruning.py
```
Please check result, `pruning.jpg` on `SiamFC`.

### STRIP
```bash
python3 ./main/test_strip.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb_strip.yaml'

python ../../tools/eval_strip.py
```
Please check result, `strip.jpg` on `SiamFC`.

## Training


export PYTHONPATH=/cheng/TAT/SiamFC

conda activate pysot

CUDA_VISIBLE_DEVICES=1 python3 ./main/test.py --config 'experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml'

CUDA_VISIBLE_DEVICES=0 python3 ./main/test.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'

CUDA_VISIBLE_DEVICES=1 python3 ./main/pruning_id.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml'


python3 ./main/pruning_test.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb_pruning.yaml'   --begin True  --Attack True

python3 ./main/eval_pruning.py

CUDA_VISIBLE_DEVICES=0 python3 ./main/test_strip.py --config 'experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb_strip.yaml'

python3 ./main/eval_strip.py
This code is based on [Stark](https://github.com/researchmm/Stark) for implementing backdoor attacks on STARK


## Install the environment
```bash
pip install -r requirements.txt 
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${STARK_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Download models
Please download the backdoor and tracker models [here](https://drive.google.com/drive/folders/1j5n30Xn2oI55EeVwP47U2hY9vU-m3aD1?usp=share_link).

Put `STARKS_ep0340.pth.tar` in `STARK/checkpoints/train/stark_s/baseline`.

Put `BACKDOOR.pth.tar` in `STARK/backdoor`.

## Testing
Please change variable `cfg.TEST.Attack` in config file `TAT/STARK/lib/config/stark_s/config.py` to decide whether to attack a tracker.

For example, you can set it as True and run
```bash
python tracking/test.py stark_s baseline --dataset otb --runid 100
```
Then, set it to False and run 
```bash
python tracking/test.py stark_s baseline --dataset otb --runid 200
```
The `runid` determines the folder where the results are stored.

Finally, run 
```bash
python tracking/analysis_results.py --dataset otb --runid 100
```
to check attack performance to trackers.

And run 
```bash
python tracking/analysis_results.py --dataset otb --runid 200
```
to check performance of trackers on clean data.



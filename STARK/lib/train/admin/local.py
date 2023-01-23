class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/cheng/Stark2/Stark-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/cheng/Stark2/Stark-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/cheng/Stark-main/pretrained_networks'
        self.lasot_dir = '/cheng/dataset/lasot/LaSOTBenchmark'
        self.got10k_dir = '/cheng/dataset/training_dataset/got10/train'
        self.lasot_lmdb_dir = '/cheng/Stark-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/cheng/Stark-main/data/got10k_lmdb'
        self.trackingnet_dir = '/cheng/dataset/training_dataset/trackingnet/TrackingNet'
        self.trackingnet_lmdb_dir = '/cheng/Stark-main/data/trackingnet_lmdb'
        self.coco_dir = '/cheng/dataset/training_dataset/coco'
        self.coco_lmdb_dir = '/cheng/Stark2/Stark-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/cheng/Stark2/Stark-main/data/vid'
        self.imagenet_lmdb_dir = '/cheng/Stark2/Stark-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

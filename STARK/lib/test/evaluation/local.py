from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/cheng/Stark2/Stark-main/data/got10k_lmdb'
    settings.got10k_path = '/cheng/dataset/training_dataset/got10'
    settings.got_packed_results_path = '.'
    settings.got_reports_path = '.'
    settings.lasot_lmdb_path = '/cheng/Stark2/Stark-main/data/lasot_lmdb'
    settings.lasot_path = '/cheng/dataset/lasot/LaSOTTesting'
    settings.network_path = '/cheng/Stark2/Stark-main/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/cheng/Stark2/Stark-main/data/nfs'
    settings.otb_path = '/cheng/pysot-1/testing_dataset/OTB100'
    settings.prj_dir = './'
    settings.result_plot_path = 'test/result_plots'
    settings.results_path = 'test/tracking_results'    # Where to store tracking results
    settings.save_dir = './'
    settings.segmentation_path = '/cheng/Stark2/Stark-main/test/segmentation_results'
    settings.tc128_path = '/cheng/Stark2/Stark-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/cheng/Stark2/Stark-main/data/trackingNet'
    settings.uav_path = '/cheng/Stark2/Stark-main/data/UAV123'
    settings.vot_path = '/cheng/Stark2/Stark-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


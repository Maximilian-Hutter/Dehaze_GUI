hparams = {
    "seed": 123,
    "gpus": 1,
    "gpu_mode": True,  #True
    "crop_size": None,
    "resume": False,
    "train_data_path": "C:/Data/dehaze/prepared/", #C:/Data/dehaze/prepared/
    "augment_data": False,
    "epochs_o_haze": 200,
    "epochs_nh_haze": 100,
    "epochs_cityscapes": 350,
    "batch_size": 1,
    "gen_lambda": 0.9,
    "color_lambda": 0.5,
    "pseudo_lambda": 0.4,
    "down_deep": False,
    "threads": 0,
    "height":640, #1280, 512, 288 niedrigste zahl = 248
    "width":360,    #720, 288, 288 solange durch 8 teilbar & >= 248
    "lr":2e-04,
    "beta1": 0.9595,
    "beta2": 0.9901,
    "mhac_filter": 16,  # 256
    "mha_filter": 32,    #16
    "num_mhablock": 4,  # 8
    "num_mhac":9, # 10
    "num_parallel_conv": 2,
    "kernel_list": [3,5,7],
    "pad_list": [4,12,24],
    "start_epoch": 0,
    "save_folder": "./weights/",
    "model_type": "Dehaze",
    "scale_factor": 1,
    "snapshots": 10,
    "pseudo_alpha": 1,
    #"pseudo_alpha": 0.6,
    #"hazy_alpha": 0.7,
    "hazy_alpha": 0.9,
    "resume_train": "./weights/9cityscapes_Dehaze.pth"
}
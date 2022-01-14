from config.absl_mock import Mock_Flag
from config.config import read_cfg_base

def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS
    FLAGS.cached_file_val= '/data1/cached_file/val_cached'
    FLAGS.cached_file=  '/data1/cached_file/cached'
    FLAGS.dataloader = 'ds_1_2_options'   #, ['ds_1_2_options', 'train_ds_options'],
    FLAGS.wandb_project_name = "distributed_training_benchmark"
    FLAGS.wandb_run_name = "Resnet18_baseline_model_fp16_cached_Prefetch_GPU_Thread_10_cls"
    FLAGS.wandb_mod = "run"

    FLAGS.Middle_layer_output = None
    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1':2,'2':1,'3':2,'4':2,'5':2}
    FLAGS.Encoder_block_channel_output = {'1':1,'2':1,'3':1,'4':1,'5':1}
    FLAGS.Middle_layer_output = None
    
    FLAGS.loss_type ="symmetrized"# asymmetrized (2 only options)
    # Moving average the weight From Online to Target Encoder Network
    FLAGS.moving_average = "schedule"# two options [fixed_value, schedule] schedule recommend from BYOL
    FLAGS.mixprecision='fp16' #['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.base_lr = 0.5

    FLAGS.non_contrast_binary_loss = "sum_symetrize_l2_loss_object_backg_add_original"#sum_symetrize_l2_loss_object_backg_add_original
    FLAGS.alpha = 1
    FLAGS.weighted_loss = 0.5
    FLAGS.resnet_depth = 18
    FLAGS.train_epochs = 50
    FLAGS.num_classes = 10

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128
    FLAGS.model_dir = "/data1/resnet_byol/resnet18/baseline_Testing_cached_fp16_10cls_Prefetch_set_GPU_Thread"
    #FLAGS.train_mode = "finetune"





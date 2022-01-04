from config.absl_mock import Mock_Flag
from config.config import read_cfg_base

def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    FLAGS.wandb_project_name = "heuristic_attention_represenation_learning_ResNet18"
    FLAGS.wandb_run_name = "restnet18_output_(7*7*512)_baseline_symloss"
    FLAGS.wandb_mod = "run"

    FLAGS.Middle_layer_output = None
    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1':2,'2':1,'3':2,'4':2,'5':2}
    FLAGS.Encoder_block_channel_output = {'1':1,'2':1,'3':1,'4':1,'5':1}
    
    FLAGS.loss_type ="symmetrized"# asymmetrized (2 only options)
    FLAGS.mixprecision='fp32' #['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.base_lr = 0.3

    FLAGS.non_contrast_binary_loss = "sum_symetrize_l2_loss_object_backg"
    FLAGS.alpha = 1
    FLAGS.weighted_loss = 0.5
    FLAGS.resnet_depth = 18
    FLAGS.train_epochs = 100
    FLAGS.num_classes = 100

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128
    FLAGS.model_dir = "/data1/share/resnet_byol/restnet18_output_(7*7*512)_baseline_symloss"
    #FLAGS.train_mode = "finetune"





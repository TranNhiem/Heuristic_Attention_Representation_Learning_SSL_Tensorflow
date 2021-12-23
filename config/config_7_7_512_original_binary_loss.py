from config.absl_mock import Mock_Flag
from config.config import read_cfg_base

def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    FLAGS.wandb_project_name = "heuristic_attention_representation_learning_v1"
    FLAGS.wandb_run_name = "MNC_resnet18_56_56_Binary_loss_7_7_original_loss_alpha_schedule_beta_0_5_stop_grad"

    FLAGS.Middle_layer_output = 2
    FLAGS.original_loss_stop_gradient = True
    FLAGS.Encoder_block_strides = {'1':2,'2':1,'3':2,'4':2,'5':2}
    FLAGS.Encoder_block_channel_output = {'1':1,'2':1,'3':1,'4':1,'5':1}

    FLAGS.base_lr = 0.3

    FLAGS.non_contrast_binary_loss = "sum_symetrize_l2_loss_object_backg_add_original"
    FLAGS.alpha = 1
    FLAGS.weighted_loss = 0.5


    FLAGS.model_dir = "/data1/share/resnet_byol/MNC_resnet18_56_56_Binary_loss_7_7_original_loss_alpha_schedule_beta_0_5_stop_grad"





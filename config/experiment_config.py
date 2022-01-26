from config.absl_mock import Mock_Flag
from config.config import read_cfg_base


def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    '''This Cache File still Under development'''

    # , ['ds_1_2_options', 'train_ds_options'],
    FLAGS.dataloader = 'ds_1_2_options'
    FLAGS.wandb_project_name = "heuristic_attention_representation_learning_Paper_correction"
    FLAGS.wandb_run_name = "Middle_hybrid_loss_(14_14_1024)_alpha_schedule_symloss"
    FLAGS.wandb_mod = "run"
    FLAGS.restore_checkpoint = True  # Restore Checkpoint or Not

    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 2}
    FLAGS.Encoder_block_channel_output = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
    FLAGS.Middle_layer_output = 4

    FLAGS.loss_type = "symmetrized"  # asymmetrized (2 only options)
    # Moving average the weight From Online to Target Encoder Network
    # two options [fixed_value, schedule] schedule recommend from BYOL
    FLAGS.moving_average = "schedule"
    # ['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.mixprecision = 'fp32'
    # , [ 'original', 'model_only', ],
    FLAGS.XLA_compiler = "original"
    FLAGS.base_lr = 0.3


    FLAGS.non_contrast_binary_loss = "sum_symetrize_l2_loss_object_backg_add_original" # sum_symetrize_l2_loss_object_backg_add_original
    FLAGS.alpha = 1
    # cosine schedule will increasing depending on training steps
    # ['cosine_schedule', 'custom_schedule'],
    FLAGS.alpha_schedule = "custom_schedule"
    FLAGS.weighted_loss = 0.3
    FLAGS.resnet_depth = 50
    FLAGS.train_epochs = 100
    FLAGS.num_classes = 1000

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128

    FLAGS.model_dir = "/data1/resnet_byol/resnet50_correction/Middle_hybrid_loss_(14_14_1024)_alpha_schedule_symloss"
    #FLAGS.train_mode = "finetune"

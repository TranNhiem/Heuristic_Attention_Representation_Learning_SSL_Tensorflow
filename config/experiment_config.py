from config.absl_mock import Mock_Flag
from config.config import read_cfg_base


def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    '''This Cache File still Under development'''

    # , ['ds_1_2_options', 'no_option'],
    #FLAGS.mask_path = "train_binary_mask_by_DRFI"
    FLAGS.dataloader = 'ds_1_2_options'
    FLAGS.wandb_project_name = "heuristic_attention_representation_learning_Paper_correction"
    FLAGS.wandb_run_name = "Baseline_(14_14_2048)_100epoch_subset"
    FLAGS.wandb_mod = "run"
    FLAGS.restore_checkpoint = True  # Restore Checkpoint or Not

    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 1}
    FLAGS.Encoder_block_channel_output = {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
    FLAGS.Middle_layer_output = None

    FLAGS.loss_type = "symmetrized"  # asymmetrized (2 only options)
    # Moving average the weight From Online to Target Encoder Network
    # two options [fixed_value, schedule] schedule recommend from BYOL
    FLAGS.moving_average = "schedule"
    # ['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.mixprecision = 'fp32'
    # , [ 'original', 'model_only', ],
    FLAGS.XLA_compiler = "original"
    FLAGS.base_lr = 0.5

    FLAGS.non_contrast_binary_loss = "baseline loss" # sum_symetrize_l2_loss_object_backg_add_original
    # cosine schedule will increasing depending on training steps
    # ['cosine_schedule', 'custom_schedule' , 'fixed'],
    FLAGS.alpha_schedule = "custom_schedule"
    FLAGS.alpha = 1
    FLAGS.weighted_loss = 0.5
    FLAGS.resnet_depth = 50
    FLAGS.train_epochs = 100
    FLAGS.num_classes = 100
    FLAGS.subset_percentage = 1.0

    FLAGS.train_batch_size = 128
    FLAGS.val_batch_size = 128

    FLAGS.model_dir = "/data1/resnet_byol/resnet50_correction/Baseline_(14_14_2048)_100epoch_subset"
    #FLAGS.train_mode = "finetune"
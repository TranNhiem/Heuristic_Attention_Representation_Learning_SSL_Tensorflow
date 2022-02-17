from config.absl_mock import Mock_Flag
from config.config import read_cfg_base


def read_cfg(mod="non_contrastive"):
    read_cfg_base(mod)
    flag = Mock_Flag()
    FLAGS = flag.FLAGS

    '''This Cache File still Under development'''
    FLAGS.num_workers = 2
    FLAGS.communication_method = "RING"  # ["NCCL", "auto", "RING"]
    FLAGS.collective_hint = False
    FLAGS.with_option = True
    FLAGS.wandb_project_name = "heuristic_attention_representation_learning_Paper_correction"
    FLAGS.wandb_run_name = "test_multi_machine"
    FLAGS.wandb_mod = "dryrun"
    FLAGS.restore_checkpoint = True  # Restore Checkpoint or Not

    FLAGS.original_loss_stop_gradient = False
    FLAGS.Encoder_block_strides = {'1': 2, '2': 1, '3': 2, '4': 2, '5': 2}
    FLAGS.Encoder_block_channel_output = {
        '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
    FLAGS.Middle_layer_output = None

    FLAGS.loss_type = "symmetrized"  # asymmetrized (2 only options)
    # Moving average the weight From Online to Target Encoder Network
    # two options [fixed_value, schedule] schedule recommend from BYOL

    FLAGS.moving_average = "schedule"
    # ['fp16', 'fp32'],  # fp32 is original precision
    FLAGS.mixprecision = 'fp32'
    FLAGS.precision_method = "API"  # "API"
    # , [ 'original', 'model_only', ],
    FLAGS.base_lr = 0.5

    # sum_symetrize_l2_loss_object_backg_add_original
    FLAGS.non_contrast_binary_loss = "sum_symetrize_l2_loss_object_backg_add_original"
    # cosine schedule will increasing depending on training steps
    # ['cosine_schedule', 'custom_schedule' , 'fixed'],
    FLAGS.alpha_schedule = "custom_schedule"
    FLAGS.alpha = 0.5
    FLAGS.weighted_loss = 0.5
    FLAGS.resnet_depth = 34
    FLAGS.train_epochs = 20
    FLAGS.num_classes = 20
    FLAGS.subset_percentage = 1.0

    FLAGS.single_machine_train_batch_size = 400
    FLAGS.single_machine_val_batch_size = 400
    FLAGS.model_dir = "./resnet_byol/resnet18_correction/test_multi_machine"
    #FLAGS.train_mode = "finetune"

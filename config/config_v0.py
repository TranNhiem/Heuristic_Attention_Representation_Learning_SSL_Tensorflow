from config.absl_mock import Mock_Flag


def read_cfg(mod="non_contrastive"):
    flags = Mock_Flag()
    base_cfg()
    wandb_set()

    if(mod == "non_contrastive"):
        non_contrastive_cfg()
    else:
        contrastive_cfg()


def base_cfg():
    flags = Mock_Flag()
    flags.DEFINE_integer(
        'IMG_height', 224,
        'image height.')

    flags.DEFINE_integer(
        'IMG_width', 224,
        'image width.')

    flags.DEFINE_float(
        'LARGE_NUM', 1e9,
        'LARGE_NUM to multiply with Logit.')

    flags.DEFINE_integer(
        'image_size', 224,
        'image size.')

    flags.DEFINE_integer(
        'SEED', 26,  # 40, 26
        'random seed use for shuffle data Generate two same image ds_one & ds_two')

    flags.DEFINE_integer(
        'SEED_data_split', 100,
        'random seed for spliting data the same for all the run with the same validation dataset.')

    flags.DEFINE_integer(
        'train_batch_size', 200,
        'Train batch_size .')

    flags.DEFINE_integer(
        'val_batch_size', 200,
        'Validaion_Batch_size.')

    flags.DEFINE_integer(
        'train_epochs', 100,
        'Number of epochs to train for.')

    flags.DEFINE_integer(
        'num_classes', 200,
        'Number of class in training data.')

    flags.DEFINE_string(
        #'train_path', "/mnt/sharefolder/Datasets/SSL_dataset/ImageNet/1K_New/ILSVRC2012_img_train",
        'train_path', '/data1/share/1K_New/train/',
        'Train dataset path.')

    flags.DEFINE_string(
        # 'val_path',"/mnt/sharefolder/Datasets/SSL_dataset/ImageNet/1K_New/val",
        'val_path', "/data1/share/1K_New/val/",
        'Validaion dataset path.')

    # Mask_folder should locate in location and same level of train folder
    flags.DEFINE_string(
        'mask_path', "train_binary_mask_by_USS",
        'Mask path.')

    flags.DEFINE_string(
        'train_label', "image_net_1k_lable.txt",
        'train_label.')

    flags.DEFINE_string(
        'val_label', "ILSVRC2012_validation_ground_truth.txt",
        'val_label.')


def wandb_set():
    flags = Mock_Flag()
    flags.DEFINE_string(
        "wandb_project_name", "heuristic_attention_representation_learning_v1",
        "set the project name for wandb."
    )
    flags.DEFINE_string(
        "wandb_run_name", "BNC_mlp_opt1_indexer_GAPool_mask_resize_cosine_lr_2_cycle",
        "set the run name for wandb."
    )
    flags.DEFINE_enum(
        'wandb_mod', 'run', ['run', 'dryrun'],
        'update the to the wandb server or not')


def Linear_Evaluation():
    flags = Mock_Flag()
    flags.DEFINE_enum(
        'linear_evaluate', 'standard', [
            'standard', 'randaug', 'cropping_randaug'],
        'How to scale the learning rate as a function of batch size.')

    flags.DEFINE_integer(
        'eval_steps', 0,
        'Number of steps to eval for. If not provided, evals over entire dataset.')
    # Configure RandAugment for validation dataset augmentation transform

    flags.DEFINE_float(
        'randaug_transform', 1,
        'Number of augmentation transformations.')

    flags.DEFINE_float(
        'randaug_magnitude', 7,
        'Number of augmentation transformations.')


def Learning_Rate_Optimizer_and_Training_Strategy():
    flags = Mock_Flag()
    # Learning Rate Strategies
    flags.DEFINE_enum(
        'lr_strategies', 'cos_annealing_restart', [
            'warmup_cos_lr', 'cos_annealing_restart', 'warmup_cos_annealing_restart'],
        'Different strategies for lr rate'
    )
    # Warmup Cosine Learning Rate Scheudle Configure
    flags.DEFINE_float(
        'base_lr', 0.5,
        'Initial learning rate per batch size of 256.')

    flags.DEFINE_integer(
        'warmup_epochs', 10,  # Configure BYOL and SimCLR
        'warmup epoch steps for Cosine Decay learning rate schedule.')

    flags.DEFINE_enum(
        'lr_rate_scaling', 'linear', ['linear', 'sqrt', 'no_scale', ],
        'How to scale the learning rate as a function of batch size.')

    #  Cosine Annelaing Restart Learning Rate Scheudle Configure
    flags.DEFINE_float(
        'number_cycles', 2.0,
        'Number of cycle for learning rate If Cycle steps is equal'
    )

    # Optimizer
    # Same the Original SimClRV2 training Configure
    '''ATTENTION'''
    flags.DEFINE_enum(

        # if Change the Optimizer please change --
        'optimizer', 'LARSW', ['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                               'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'],
        'How to scale the learning rate as a function of batch size.')

    flags.DEFINE_enum(
        # Same the Original SimClRV2 training Configure
        # 1. original for ['Adam', 'SGD', 'LARS']
        # 2.optimizer_weight_decay for ['AdamW', 'SGDW', 'LARSW']
        # 3. optimizer_GD fir  ['AdamGC', 'SGDGC', 'LARSGC']
        # 4. optimizer_W_GD for ['AdamW_GC', 'SGDW_GC', 'LARSW_GC']

        'optimizer_type', 'optimizer_weight_decay', [
            'original', 'optimizer_weight_decay', 'optimizer_GD', 'optimizer_W_GD'],
        'Optimizer type corresponding to Configure of optimizer')

    flags.DEFINE_float(
        'momentum', 0.9,
        'Momentum parameter.')

    flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')


def Encoder():
    flags = Mock_Flag()
    flags.DEFINE_boolean(
        'global_bn', True,
        'Whether to aggregate BN statistics across distributed cores.')

    flags.DEFINE_float(
        'batch_norm_decay', 0.9,  # Checkout BN decay concept
        'Batch norm decay parameter.')

    flags.DEFINE_integer(
        'width_multiplier', 1,
        'Multiplier to change width of network.')

    flags.DEFINE_integer(
        'resnet_depth', 50,
        'Depth of ResNet.')

    flags.DEFINE_float(
        'sk_ratio', 0.,
        'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')
    flags.DEFINE_enum(
        "Middle_layer_output", 0, [0, 1, 2, 3, 4, 5],
        "Get the feature map from middle layer,0 is mean don't get the middle layer feature map"
    )

    flags.DEFINE_float(
        'se_ratio', 0.,
        'If it is bigger than 0, it will enable SE.')


def Projection_and_Prediction_head():

    flags = Mock_Flag()

    flags.DEFINE_enum(
        'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
        'How the head projection is done.')

    # Projection & Prediction head  (Consideration the project out dim smaller than Represenation)

    flags.DEFINE_integer(
        'proj_out_dim', 256,
        'Number of head projection dimension.')

    flags.DEFINE_integer(
        'prediction_out_dim', 256,
        'Number of head projection dimension.')

    flags.DEFINE_boolean(
        'reduce_linear_dimention', True,  # Consider use it when Project head layers > 2
        'Reduce the parameter of Projection in middel layers.')
    flags.DEFINE_integer(
        'up_scale', 4096,  # scaling the Encoder output 2048 --> 4096
        'Upscale the Dense Unit of Non-Contrastive Framework')

    flags.DEFINE_boolean(
        'non_contrastive', True,  # Consider use it when Project head layers > 2
        'Using for upscaling the first layers of MLP == upscale value')

    flags.DEFINE_integer(
        'num_proj_layers', 3,
        'Number of non-linear head layers.')

    flags.DEFINE_integer(
        'ft_proj_selector', 0,
        'Which layer of the projection head to use during fine-tuning. '
        '0 means no projection head, and -1 means the final layer.')

    flags.DEFINE_float(
        'temperature', 0.3,
        'Temperature parameter for contrastive loss.')

    flags.DEFINE_boolean(
        'hidden_norm', True,
        'L2 Normalization Vector representation.')

    flags.DEFINE_enum(
        'downsample_mod', 'space_to_depth', [
            'space_to_depth', 'maxpooling', 'averagepooling'],
        'How the head upsample is done.')

    flags.DEFINE_boolean(
        'feature_upsample', False,
        'encoder out put do the upsample or mask do the downsample'
    )


def Configure_Model_Training():
    # Self-Supervised training and Supervised training mode
    flags = Mock_Flag()
    flags.DEFINE_enum(
        'mode', 'train', ['train', 'eval', 'train_then_eval'],
        'Whether to perform training or evaluation.')

    flags.DEFINE_enum(
        'train_mode', 'pretrain', ['pretrain', 'finetune'],
        'The train mode controls different objectives and trainable components.')

    flags.DEFINE_boolean('lineareval_while_pretraining', True,
                         'Whether to finetune supervised head while pretraining.')

    flags.DEFINE_enum(
        'aggregate_loss', 'contrastive_supervised', [
            'contrastive', 'contrastive_supervised', ],
        'Consideration update Model with One Contrastive or sum up and (Contrastive + Supervised Loss).')

    flags.DEFINE_enum(
        'non_contrast_binary_loss', 'sum_symetrize_l2_loss_object_backg', ["byol_harry_loss",
                                                                           'Original_loss_add_contrast_level_object', 'sum_symetrize_l2_loss_object_backg', 'original_add_backgroud'],
        'Consideration update Model with One Contrastive or sum up and (Contrastive + Supervised Loss).')

    flags.DEFINE_float(
        # Alpha Weighted loss (Objec & Background) [binary_mask_nt_xent_object_backgroud_sum_loss]
        'alpha', 0.99,
        'Alpha value is configuration the weighted of Object and Background in Model Total Loss.'
    )
    flags.DEFINE_float(
        # Weighted loss is the scaling term between  [weighted_loss]*Binary & [1-weighted_loss]*original contrastive loss)
        'weighted_loss', 0.8,
        'weighted_loss value is configuration the weighted of original and Binary contrastive loss.'
    )
    # Fine Tuning configure

    flags.DEFINE_boolean(
        'zero_init_logits_layer', False,
        'If True, zero initialize layers after avg_pool for supervised learning.')

    flags.DEFINE_integer(
        'fine_tune_after_block', -1,
        'The layers after which block that we will fine-tune. -1 means fine-tuning '
        'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
        'just the linear head.')


def Configure_Saving_and_Restore_Model():
    # Saving Model
    flags = Mock_Flag()
    flags.DEFINE_string(
        'model_dir', "./model_ckpt/resnet_byol/cosine_restart/",
        'Model directory for training.')

    flags.DEFINE_integer(
        'keep_hub_module_max', 1,
        'Maximum number of Hub modules to keep.')

    flags.DEFINE_integer(
        'keep_checkpoint_max', 5,
        'Maximum number of checkpoints to keep.')

    # Loading Model

    # Restore model weights only, but not global step and optimizer states
    flags.DEFINE_string(
        'checkpoint', None,
        'Loading from the given checkpoint for fine-tuning if a finetuning '
        'checkpoint does not already exist in model_dir.')

    flags.DEFINE_integer(
        'checkpoint_epochs', 1,
        'Number of epochs between checkpoints/summaries.')

    flags.DEFINE_integer(
        'checkpoint_steps', 10,
        'Number of steps between checkpoints/summaries. If provided, overrides '
        'checkpoint_epochs.')


def visualization():
    flags = Mock_Flag()
    flags.DEFINE_boolean("visualize",
                         False, "visualize the feature map or not"
                         )
    flags.DEFINE_integer("visualize_epoch",
                         1, "Number of every epoch to save the feature map"
                         )
    flags.DEFINE_string("visualize_dir",
                        "/visualize", "path of the visualize feature map saved"
                        )


def non_contrastive_cfg():
    Linear_Evaluation()
    Learning_Rate_Optimizer_and_Training_Strategy()
    Encoder()
    Projection_and_Prediction_head()
    Configure_Model_Training()
    Configure_Saving_and_Restore_Model()
    visualization()


def contrastive_cfg():

    flags = Mock_Flag()
    # ------------------------------------------
    # Define for Linear Evaluation
    # ------------------------------------------
    flags.DEFINE_enum(
        'linear_evaluate', 'standard', [
            'standard', 'randaug', 'cropping_randaug'],
        'How to scale the learning rate as a function of batch size.')

    flags.DEFINE_integer(
        'eval_steps', 0,
        'Number of steps to eval for. If not provided, evals over entire dataset.')

    flags.DEFINE_float(
        'randaug_transform', 1,
        'Number of augmentation transformations.')

    flags.DEFINE_float(
        'randaug_magnitude', 7,
        'Number of augmentation transformations.')
    # ---------------------------------------------------
    # Define for Learning Rate Optimizer
    # ---------------------------------------------------
    # Learning Rate Scheudle

    flags.DEFINE_float(
        'base_lr', 0.3,
        'Initial learning rate per batch size of 256.')

    flags.DEFINE_integer(
        'warmup_epochs', 10,  # Configure BYOL and SimCLR
        'warmup epoch steps for Cosine Decay learning rate schedule.')

    flags.DEFINE_enum(
        'lr_rate_scaling', 'linear', ['linear', 'sqrt', 'no_scale', ],
        'How to scale the learning rate as a function of batch size.')
    # Optimizer
    # Same the Original SimClRV2 training Configure
    '''ATTENTION'''
    flags.DEFINE_enum(

        # if Change the Optimizer please change --
        'optimizer', 'LARSW', ['Adam', 'SGD', 'LARS', 'AdamW', 'SGDW', 'LARSW',
                               'AdamGC', 'SGDGC', 'LARSGC', 'AdamW_GC', 'SGDW_GC', 'LARSW_GC'],
        'How to scale the learning rate as a function of batch size.')

    flags.DEFINE_enum(
        # Same the Original SimClRV2 training Configure
        # 1. original for ['Adam', 'SGD', 'LARS']
        # 2.optimizer_weight_decay for ['AdamW', 'SGDW', 'LARSW']
        # 3. optimizer_GD fir  ['AdamGC', 'SGDGC', 'LARSGC']
        # 4. optimizer_W_GD for ['AdamW_GC', 'SGDW_GC', 'LARSW_GC']

        'optimizer_type', 'optimizer_weight_decay', [
            'original', 'optimizer_weight_decay', 'optimizer_GD', 'optimizer_W_GD'],
        'Optimizer type corresponding to Configure of optimizer')

    flags.DEFINE_float(
        'momentum', 0.9,
        'Momentum parameter.')

    flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

    # ------------------------------------------------------------------------------
    # Configure for Encoder - Projection Head, Linear Evaluation Architecture
    # ------------------------------------------------------------------------------
    # Encoder Configure

    flags.DEFINE_boolean(
        'global_bn', True,
        'Whether to aggregate BN statistics across distributed cores.')

    flags.DEFINE_float(
        'batch_norm_decay', 0.9,  # Checkout BN decay concept
        'Batch norm decay parameter.')

    flags.DEFINE_integer(
        'width_multiplier', 1,
        'Multiplier to change width of network.')

    flags.DEFINE_integer(
        'resnet_depth', 50,
        'Depth of ResNet.')

    flags.DEFINE_float(
        'sk_ratio', 0.,
        'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

    flags.DEFINE_float(
        'se_ratio', 0.,
        'If it is bigger than 0, it will enable SE.')

    # Projection Head

    flags.DEFINE_enum(
        'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
        'How the head projection is done.')

    flags.DEFINE_integer(
        'proj_out_dim', 128,
        'Number of head projection dimension.')

    flags.DEFINE_integer(
        'num_proj_layers', 3,
        'Number of non-linear head layers.')

    flags.DEFINE_integer(
        'ft_proj_selector', 0,
        'Which layer of the projection head to use during fine-tuning. '
        '0 means no projection head, and -1 means the final layer.')

    flags.DEFINE_float(
        'temperature', 0.1,
        'Temperature parameter for contrastive loss.')

    flags.DEFINE_boolean(
        'hidden_norm', True,
        'L2 Normalization Vector representation.')

    # -------------------------------------------------------------------
    # Configure Model Training -- Loss Function Implementation --- Evaluation -- FineTuning --
    # -------------------------------------------------------------------

    # Configure Model Training [BaseLine Model]

    flags.DEFINE_enum(
        'mode', 'train', ['train', 'eval', 'train_then_eval'],
        'Whether to perform training or evaluation.')

    flags.DEFINE_enum(
        'train_mode', 'pretrain', ['pretrain', 'finetune'],
        'The train mode controls different objectives and trainable components.')

    flags.DEFINE_boolean('lineareval_while_pretraining', True,
                         'Whether to finetune supervised head while pretraining.')

    flags.DEFINE_enum(
        'aggregate_loss', 'contrastive_supervised', [
            'contrastive', 'contrastive_supervised', ],
        'Consideration update Model with One Contrastive or sum up and (Contrastive + Supervised Loss).')

    # Configure Model Training [Contrastive Binary Loss]

    flags.DEFINE_float(
        # Alpha Weighted loss (Objec & Background) [binary_mask_nt_xent_object_backgroud_sum_loss]
        'alpha', 0.8,
        'Alpha value is configuration the weighted of Object and Background in Model Total Loss.'
    )

    flags.DEFINE_float(
        # Weighted loss is the scaling term between  [weighted_loss]*Binary & [1-weighted_loss]*original contrastive loss)
        'weighted_loss', 0.7,
        'weighted_loss value is configuration the weighted of original and Binary contrastive loss.'
    )

    flags.DEFINE_enum(
        'contrast_binary_loss', 'sum_contrast_obj_back',
        # 4 Options Loss for training.
        [
            # two version binary_mask_nt_xent_object_backgroud_sum_loss, binary_mask_nt_xent_object_backgroud_sum_loss_v1
            "sum_contrast_obj_back",
            "only_object",  # binary_mask_nt_xent_only_Object_loss
            # Concatenate (Object + Background feature together)
            "original_contrast_add_backgroud_object",
            # nt_xent_symetrize_loss_object_level_whole_image_contrast
            "Original_loss_add_contrast_level_object",
        ],
        # sum_contrast_obj_back is Sum-up contrastive loss from backgroud and Object with scaling alpha
        #
        'Contrast binary Framework consider three different LOSSES')

    flags.DEFINE_enum(
        'loss_options', 'loss_v0',
        ['loss_v0', 'loss_v1'],
        "Option for chossing loss version [V0]--> Original simclr loss [V1] --> Custom build design loss"
    )

    # Fine Tuning configure

    flags.DEFINE_boolean(
        'zero_init_logits_layer', False,
        'If True, zero initialize layers after avg_pool for supervised learning.')

    flags.DEFINE_integer(
        'fine_tune_after_block', -1,
        'The layers after which block that we will fine-tune. -1 means fine-tuning '
        'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
        'just the linear head.')

    # -------------------------------------------------------------------
    # Configure Saving and Restore Model
    # -------------------------------------------------------------------

    # Saving Model
    flags.DEFINE_string(
        'model_dir', "./model_ckpt/simclrResNet/",
        'Model directory for training.')

    flags.DEFINE_integer(
        'keep_hub_module_max', 1,
        'Maximum number of Hub modules to keep.')

    flags.DEFINE_integer(
        'keep_checkpoint_max', 5,
        'Maximum number of checkpoints to keep.')

    # Loading Model

    # Restore model weights only, but not global step and optimizer states

    flags.DEFINE_string(
        'checkpoint', None,
        'Loading from the given checkpoint for fine-tuning if a finetuning '
        'checkpoint does not already exist in model_dir.')

    flags.DEFINE_integer(
        'checkpoint_epochs', 10,
        'Number of epochs between checkpoints/summaries.')

    flags.DEFINE_integer(
        'checkpoint_steps', 10,
        'Number of steps between checkpoints/summaries. If provided, overrides '
        'checkpoint_epochs.')

    # Helper function to save and resore model.

    # Helper function to save and resore model.

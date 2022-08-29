# HARL - Heuristic Attention Representation Learning for Self-Supervised Pretraining

<span style="color: red"><strong> </strong></span> We have released a Pytorch Lightning implementation; (along with checkpoints)</a>.

<span style="color: red"><strong> </strong></span> Colabs for <a href=""> HARL framework </a> are added, see <a href="">here</a>.

<div align="center">
  <img width="50%" alt="HARL Framework Illustration" src="images/HARL.GIF">
</div>
<div align="center">
  An illustration of HARL Framework (from <a href="https://www.hh-ri.com/2022/05/30/heuristic-attention-representation-learning-for-self-supervised-pretraining/">our blog here</a>).
</div>

## HARL Pre-trained models  

We open-sourced the pretrained models here, corresponding to those in Table 1 of the <a href="https://pdfs.semanticscholar.org/0040/f14dac94ea4fd96072f2b98686b57dde2dfb.pdf?_ga=2.79292561.484170449.1659857629-2034857879.1659857629">HARL</a> paper:
These checkpoints are stored in Google Drive Storage:

|   Depth | Width   | SK    |   Param (M)  |    Linear eval |   Supervised |
|--------:|--------:|------:|--------:|-------------:|--------------:|
|      50 | 1X      | False |    24 |        74.0 |          76.6 |


The pre-trained models (base network with linear classifier layer) can be found below.

|                             Model checkpoint and hub-module                             |     ImageNet Top-1     |
|-----------------------------------------------------------------------------------------|------------------------|
|[ResNet50 (1x)](https://drive.google.com/drive/folders/1oNkxwA-VixlnUBGgxVeHrcPcDmKHeND1?usp=sharing) |          74.0          |


## Enviroment setup

Our code can also run on a *single* GPU or *multi-GPUs* GPUs.

The code is compatible with different Pytorch lightning versions . See requirements.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Pretraining

The following command can be used to pretrain a ResNet-50 on ImageNet (which reflects the default hyperparameters in our paper):

```
The instructions will update soon :)
```

A batch size of 4096 requires 8 A100 GPUs with 80G of VRAM. 1000 epochs takes around 149 hour. Note that learning rate of 0.2 with `learning_rate_scaling=linear` is equivalent to that of 0.075 with `learning_rate_scaling=sqrt` when the batch size is 4096. However, using sqrt scaling allows it to train better when smaller batch size is used.

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

```
The instructions will update soon :)

```

For fine-tuning a linear head on ImageNet using GPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:

```
The instructions will update soon :)
```

As a reference, the above runs on ImageNet should give you around 64.5% accuracy.

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

```
The instructions will update soon :)
```

Set the `checkpoint` to those that are only pre-trained but not fine-tuned. Given that SimCLRv1 checkpoints do not contain projection head, it is recommended to run with SimCLRv2 checkpoints (you can still run with SimCLRv1 checkpoints, but `variable_schema` needs to exclude `head`). The `num_proj_layers` and `ft_proj_selector` need to be adjusted accordingly following SimCLRv2 paper to obtain best performances.

## Other resources

### Our *offical* implementations in Different Frameworks

(Feel free to share your implementation by creating an issue)

Implementations in Pytorch Lightning:
* [Official Implementation](https://github.com/TranNhiem/Heuristic_Attention_Represenation_Learning_SSL_Pytorch)

## Cite
[HARL paper](https://pdfs.semanticscholar.org/0040/f14dac94ea4fd96072f2b98686b57dde2dfb.pdf?_ga=2.79292561.484170449.1659857629-2034857879.1659857629):

```
@article{Tran2022HeuristicAR,
  title={Heuristic Attention Representation Learning for Self-Supervised Pretraining},
  author={Van-Nhiem Tran and Shenxiu Liu and Yung-hui Li and Jia-Ching Wang},
  journal={Sensors (Basel, Switzerland)},
  year={2022},
  volume={22}
}
```





import argparse

parser = argparse.ArgumentParser(description='CLTR')

parser.add_argument('--dataset', type=str, default='jhu',
                    help='choice train dataset')

parser.add_argument('--save_path', type=str, default='save_file/A_ddp',
                    help='save checkpoint directory')
parser.add_argument('--workers', type=int, default=2,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')
parser.add_argument('--epochs', type=int, default=5000,
                    help='end epoch for training')
parser.add_argument('--pre', type=str, default=None,
                    help='pre-trained model directory')

parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--crop_size', type=int, default=256,
                    help='crop size for training')
parser.add_argument('--lr_step', type=int, default=1200,
                    help='lr_step')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='0,1',
                    help='gpu id')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')

parser.add_argument('--save',  action='store_true',
                    help='save the file')
parser.add_argument('--scale_aug', action='store_true',
                    help='using the scale augmentation')
parser.add_argument('--scale_type', type=int, default=0,
                    help='scale type')
parser.add_argument('--scale_p', type=float, default=0.3,
                    help='probability of scaling')
parser.add_argument('--gray_aug', action='store_true',
                    help='using the gray augmentation')
parser.add_argument('--gray_p', type=float, default=0.1,
                    help='probability of gray')
parser.add_argument('--test_patch', action='store_true',
                    help='true test_patch ')
parser.add_argument('--channel_point', type=int, default=3,
                    help='number of boxes')
parser.add_argument('--num_patch', type=int, default=1,
                    help='number of patches')
parser.add_argument('--min_num', type=int, default=-1,
                    help='min_num')
parser.add_argument('--num_knn', type=int, default=4,
                    help='number of knn')
parser.add_argument('--test_per_epoch', type=int, default=20,
                    help='test_per_epoch')
parser.add_argument('--threshold', type=float, default=0.35,
                    help='threshold to filter the negative points')

# video demo
parser.add_argument('--video_path', type=str, default='./video_demo/1.mp4',
                    help='input video path ')

# distributed training parameters
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local local_rank')

parser.add_argument('--lr_backbone', default=1e-4, type=float)
parser.add_argument('--lr_drop', default=40, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=500, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation, not used
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")

# * Matcher,some parameters are not used here
parser.add_argument('--set_cost_class', default=2, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_point', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost") # not used

# * Loss coefficients, some parameters are not used here
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--cls_loss_coef', default=2, type=float)
parser.add_argument('--count_loss_coef', default=2, type=float)
parser.add_argument('--point_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--focal_alpha', default=0.25, type=float)

# dataset parameters, not used
parser.add_argument('--dataset_file', default='crowd_data')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')

parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env:// ', help='url used to set up distributed training')
parser.add_argument('--master_port', default=29501, type=int,
                    help='master_port')

args = parser.parse_args()
return_args = parser.parse_args()

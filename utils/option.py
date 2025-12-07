import argparse

def init():
    # ************************************************************************training parameters************************************************************************

    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--train_csv_path', type=str,default="/home/cjg/workSpace/AVA_image_sort_v2/ICTA2Net/data/train/",
                        help='directory to csv_folder')
    parser.add_argument('--test_csv_path', type=str,default="/home/cjg/workSpace/AVA_image_sort_v2/ICTA2Net/data/test/",
                        help='directory to csv_folder')
    parser.add_argument('--image_path_G', type=str,default="/mnt/group_temp/cjg/Temp/FiveK/group_temp_imgname_tint=0",
                        help='directory to imag')
    parser.add_argument('--image_path_H', type=str,default="/mnt/group_temp/cjg/Temp/PPR10k/group_temp_imgname_tint=0",
                        help='directory to imag')
    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--pre_weight', type=str, default='None',
                        help='directory to pretrain model')

    parser.add_argument('--init_lr', type=int, default=1e-4, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=10, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int,default=24,help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help ='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')
    parser.add_argument('--model_save_path', type=str, default='/home/cjg/workSpace/AVA_image_sort_v2/ICTA2Net/weight', help='which gpu to use')
    parser.add_argument('--resume', type=bool, default=False, help='whether resume training')
    parser.add_argument('--checkpoint_path', type=str, default="/home/cjg/workSpace/AVA_image_sort_v2/ICTA2Net/weight/ICTA2Net-best---.pth", help='checkpoint_path')
    parser.add_argument('--ablate_text', type=bool, default=False, help='whether ablate text')

    # ************************************************************************model parameters************************************************************************
    # model construction related parameters
    parser.add_argument('--clip_pretrain', type=str, default='/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/DETRIS/pretrain/ViT-B-16.pt',
                        help='Path to CLIP pretrained model')
    parser.add_argument('--dino_pretrain', type=str, default='/home/cjg/workSpace/AVA_image_sort_v2/Multi_modal_AWB/DETRIS/pretrain/dinov2_vitb14_reg4_pretrain.pth',
                        help='Path to DINO pretrained model')
    parser.add_argument('--dino_name', type=str, default='dino-base', help='Name of DINO model')
    parser.add_argument('--word_len', type=int, default=77, help='Length of input words')
    parser.add_argument('--input_size', type=int, default=448, help='Input image size')
    parser.add_argument('--txtual_adapter_layer', nargs='+', type=int, default=[1, 3, 5, 7, 9, 11],
                        help='Textual adapter layers')
    parser.add_argument('--txt_adapter_dim', type=int, default=64, help='Textual adapter dimension')
    parser.add_argument('--ladder_dim', type=int, default=128, help='Ladder dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of heads in multi - head attention')
    parser.add_argument('--dino_layers', type=int, default=12, help='Number of DINO layers')
    parser.add_argument('--output_dinov2', nargs='+', type=int, default=[4, 8],
                        help='Output DINO layers')
    parser.add_argument('--visual_adapter_layer', nargs='+', type=int, default=[1, 3, 5, 7, 9, 11],
                        help='Visual adapter layers')
    parser.add_argument('--visual_adapter_dim', type=int, default=128,
                        help='Visual adapter dimension')


    # add Neck module related parameters
    parser.add_argument('--fpn_in', nargs='+', type=int, default=[768, 768, 768],
                        help='Input channels for FPN in Neck module')
    parser.add_argument('--fpn_out', nargs='+', type=int, default=[256, 512, 1024],
                        help='Output channels for FPN in Neck module')
    parser.add_argument('--stride', nargs='+', type=int, default=[1, 1, 1],
                        help='Stride for Neck module')

    # add Decoder module related parameters
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in Decoder module')
    parser.add_argument('--vis_dim', type=int, default=512,
                        help='Visual dimension in Decoder module')
    parser.add_argument('--num_head', type=int, default=8,
                        help='Number of heads in Decoder module')
    parser.add_argument('--dim_ffn', type=int, default=512,
                        help='Feed-forward network dimension in Decoder module')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in Decoder module')
    parser.add_argument('--intermediate', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to return intermediate outputs in Decoder module')

    # add training settings related parameters
    parser.add_argument('--lr_multi', type=float, default=1,
                        help='Learning rate multiplier for backbone')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Base learning rate')

    # ************************************************************************overlock模型参数************************************************************************
    parser.add_argument('--model', default='overlock_xt', type=str, help='Name of model to create')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained model')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')
    # add gradient checkpointing related parameters
    parser.add_argument('--grad_checkpoint', action='store_true', default=False, help='Using gradient checkpointing for saving GPU memory')
    parser.add_argument('--ckpt_stg', default=[0, 0, 0, 0], type=int, nargs='+', help='Stage for using grad checkpoint')

    args = parser.parse_args()
    return args

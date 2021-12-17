import argparse
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    parser.add_argument('--video_path',
                        default=None,
                        type=Path,
                        help='Directory path of videos')
    parser.add_argument('--n_classes',
                        default=2,
                        type=int,
                        help=
                        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    parser.add_argument('--val_annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--pretrain_path',
                        default=None,
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--model', 
                        default='', 
                        type=str,
                        help='Model file path')
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', 
                        default=0.9, 
                        type=float, 
                        help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='plateau',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument('--manual_seed',
                        default='12',
                        type=int,
                        help='To set seed manually')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='If true, do not print model')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args

import argparse
import torch
import uuid

from paint_it import main
from sd import StableDiffusion


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--obj_path', type=str)
    parser.add_argument('--prompt', type=str, default='a car')

    # model
    parser.add_argument('--decay', type=float, default=0)  # weight decay
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--decay_step', type=int, default=100)

    # training
    parser.add_argument('--sd_max_grad_norm', type=float, default=10.0)
    parser.add_argument('--n_iter', type=int, default=1500)  # can be increased
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--sd_min', type=float, default=0.2)
    parser.add_argument('--sd_max', type=float, default=0.98)
    parser.add_argument('--sd_min_l', type=float, default=0.2)
    parser.add_argument('--sd_min_r', type=float, default=0.3)
    parser.add_argument('--sd_max_l', type=float, default=0.5)
    parser.add_argument('--sd_max_r', type=float, default=0.98)
    parser.add_argument('--bg', type=float, default=0.25)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sd_minmax_anneal', type=eval, default=True, choices=[True, False])
    parser.add_argument('--n_view', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--env_scale', type=float, default=2.0)
    parser.add_argument('--envmap', type=str, default='data/irrmaps/mud_road_puresky_4k.hdr')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--gd_scale', type=int, default=100)
    parser.add_argument('--learn_light', type=eval, default=True, choices=[True, False])

    args = parser.parse_args()
    args.kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
    args.kd_max = [1.0, 1.0, 1.0, 1.0]
    args.ks_min = [0.0, 0.08, 0.0]  # Limits for ks
    args.ks_max = [1.0, 1.0, 1.0]
    args.nrm_min = [-0.1, -0.1, 0.0]  # Limits for normal map
    args.nrm_max = [0.1, 0.1, 1.0]
    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load stable-diffusion model
    guidance = StableDiffusion(device, min=args.sd_min, max=args.sd_max)
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False

    obj_id = str(uuid.uuid4()).replace('-', '')

    args.exp_name = obj_id
    args.obj_id = obj_id

    args.identity = args.prompt
    main(args, guidance)

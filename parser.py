import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    # ncl
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='amazon-sports-outdoors')
    parser.add_argument('--model', type=str, default='mclf')
    parser.add_argument('--imagek', type=int, default=11)
    parser.add_argument('--textk', type=int, default=11)
    parser.add_argument('--text', type=int, default=1)
    parser.add_argument('--image', type=int, default=1)
    parser.add_argument('--cluster', type=int, default=1)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--image_weight', type=float, default=0.2)
    parser.add_argument('--text_weight', type=float, default=1)



    return parser.parse_args()
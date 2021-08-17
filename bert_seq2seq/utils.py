import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/configs.yml',
        help='Path to the configration yaml file.'
    )

    return parser.parse_args()
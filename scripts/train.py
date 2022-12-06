import argparse

from loguru import logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dn', '--data_name', type=str, help='Dataset name to use', required=True)
    parser.add_argument('-s', '--size', type=int, help='Image size to train', default=64)
    parser.add_argument('-e', '--expn', type=str, help='Experiment name (basename of output directory)', default=None)
    args = parser.parse_args()
    return args


@logger.catch
def main():
    args = vars(get_args())
    print(args)
    assert args['size'] in {64, 128, 256}, (
        'Image size should be one of [64, 128, 256]\n'
        'If you want to train other size, create config file named {iamge_size}_model.yml in diffae/cfg and modify it.'
    )

    module = __import__('diffae', fromlist=['DiffusionAutoEncodersInterface'])
    interface_cls = getattr(module, 'DiffusionAutoEncodersInterface')
    interface = interface_cls(args, mode='train')

    interface.train()


if __name__ == '__main__':
    main()

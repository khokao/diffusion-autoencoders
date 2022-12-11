import argparse

from loguru import logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    return args


@logger.catch
def main():
    args = vars(get_args())
    print(args)

    module = __import__('diffae', fromlist=['DiffusionAutoEncodersInterface'])
    interface_cls = getattr(module, 'DiffusionAutoEncodersInterface')
    interface = interface_cls(args, mode='clf_test')

    interface.clf_test()


if __name__ == '__main__':
    main()

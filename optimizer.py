import argparse

import logging
import logging.config

import opt_rest
import opt_utils

logger = None
args = None


def opt_main():
    global args
    args = parse_arguments()

    # load config file
    cfg = opt_utils.read_yaml(args.config_path)

    # create neccessary directories 
    opt_utils.create_dirs(cfg.get('directories', ['data/nn_training_data.csv', 'data/lr_training_data.csv']).values())

    # create logger from provided config
    global logger
    logger = create_logger(cfg)

    # init service and run optimizer REST
    opt_rest.init_service(cfg)
    opt_rest.app.run(debug=True,
                     host=args.host,
                     port=args.port)
                     

def create_logger(config):
    try:
        logging.config.dictConfig(config.get('logging'))
        logger = logging.getLogger('optimizer')
    except Exception as e:
        print(f'ERROR: Cannot process configuration file "{args.config_path}": {e}')
    else:
        logger.info('Optimizer initialized successfully')
        return logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='MiCADO component to realize optimization of scaling decisions')
    parser.add_argument('--cfg',
                        dest='config_path',
                        default='./config/config.yaml',
                        help='path to configuration file')

    parser.add_argument('--host',
                        type=str,
                        default='127.0.0.1',
                        help='host to bind service to')
    parser.add_argument('--port',
                        type=int,
                        default='5000',
                        help='port to bind service to')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt_main()
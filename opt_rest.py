from flask import Flask, jsonify, request, send_file

from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils


app = Flask(__name__)

logger = None
conf = None

constants = {}
sample_number = 0
vm_number_prev = 0

def init_service(cfg):
    global logger
    logger = logging.getLogger('optimizer')

    global conf
    conf = opt_config.OptimizerConfig(cfg, 'optimizer')


@app.route('/optimizer/init', methods=['POST'])
def init():
    global logger
    logger.debug('Loading constants from file...')
    constants_yaml = request.stream.read()
    if not constants_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        global constants
        constants = yaml.safe_load(constants_yaml).get('constants')
        logger.debug(f'Constants received: {constants}')

        logger.debug('Saving constants...')
        opt_utils.write_yaml(conf.constants_filename, constants)
        logger.debug('Constants saved to "data/constants.yaml"')
        logger.debug('Preparing database for training data...')
        # input_metrics+timestamp is redundant!
        input_metrics = [metric.get('name')
                         for metric in constants.get('input_metrics')]
        target_metrics = [metric.get('name')
                          for metric in constants.get('target_metrics')]
        timestamp_col = ['timestamp']
        vm_cols = ['vm_number', 'vm_number\'', 'vm_number_diff']

        logger.debug('Creating a .csv file for neural network...')
        opt_utils.persist_data(
            conf.nn_filename, timestamp_col+input_metrics+target_metrics, 'w')
        logger.debug('File created')

        logger.debug('Creating a .csv file for linear regression...')
        opt_utils.persist_data(
            conf.lr_filename, timestamp_col+input_metrics+vm_cols, 'w')
        logger.debug('File created')

        logger.info('Optimizer REST initialized successfully ')
    return jsonify('OK'), 200


@app.route('/optimizer/training_data', methods=['GET', 'POST'])
def training_data():
    global logger
    if request.method == 'GET':
        logger.info('Training_data requested')
        logger.debug(
            'Zipping neural network and linear regression training data files...')
        zip_file = conf.training_data_filename
        opt_utils.zip_files([conf.nn_filename, conf.lr_filename], zip_file)
        logger.debug('Files zipped')
        logger.info('Sending zipped training data...')
        return send_file(zip_file,
                         mimetype='zip',
                         attachment_filename=zip_file,
                         as_attachment=True)


@app.route('/optimizer/sample', methods=['POST'])
def sample():
    global logger
    global constants

    logger.debug('Loading training sample...')
    sample_yaml = request.stream.read()
    if not sample_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        sample = yaml.safe_load(sample_yaml)
        logger.debug(f'New sample received: {sample}')
        global sample_number
        sample_number += 1

        logger.debug('Gaining sample data...')
        input_metrics = [metric.get('value')
                         for metric in sample.get('sample').get('input_metrics')]
        target_metrics = [metric.get('value')
                          for metric in sample.get('sample').get('target_metrics')]
        vm_number = sample.get('sample').get('vm_number')
        timestamp_col = [sample.get('sample').get('timestamp')]
        logger.debug('Sample data gained')

        # timestamp+input_metrics is also redundant here!
        logger.debug(
            'Calculating difference between previous and current VM number')
        global vm_number_prev
        vm_number_diff = vm_number-vm_number_prev \
            if vm_number is not None and vm_number_prev is not None \
            else None
        vm_vals = [vm_number, vm_number_prev, vm_number_diff]

        logger.debug(
            'Saving timestamp, input and target metrics to neural network data file...')
        # redundancy again
        opt_utils.persist_data(conf.nn_filename,
                               timestamp_col+input_metrics+target_metrics, 'a')
        logger.debug('Data saved')

        logger.debug(
            'Saving timestamp, input metrics and VM number related data to linear regression data file...')
        opt_utils.persist_data(conf.lr_filename,
                               timestamp_col+input_metrics+vm_vals, 'a')
        logger.debug('Data saved')

        vm_number_prev = vm_number
        logger.info('Sample received and processed.')

    return jsonify('OK'), 200


def advice_msg(valid=False,phase='training',vmnumber=0,errmsg=None,confident=0):
  if valid:
    return jsonify(dict(valid=valid,phase=phase,vmnumber=vmnumber,confident=confident,errmsg='')), 200
  else:
    return jsonify(dict(valid=valid,phase=phase,vmnumber=vmnumber,confident=confident,errmsg=errmsg)), 400

@app.route('/optimizer/advice', methods=['GET'])
def get_advice():
    global logger
    global constants
    global sample_number

    logger.debug('Checking phase...')
    if (sample_number == 0):
        msg = 'There are no training samples yet.'
        logger.error(msg)
        return advice_msg(valid=False,errmsg=msg)
    elif (sample_number < constants['training_samples_required']):
        logger.debug('PRETRAINING PHASE')
        if (sample_number == 1):
            logger.info('PRETRAINING PHASE - 1st call')
            return advice_msg(valid=True,
                              vmnumber=int((constants.get('max_vm_number')+constants.get('min_vm_number'))/2))
        else:
            logger.info('Collecting samples to get training started...')
            return advice_msg(valid=False,
                              errmsg='Not implemented yet.')
    else:
        logger.debug('TRAINING PHASE')
        #call ml part
        import random
        return advice_msg(valid=False,
                          vmnumber=random.randint(constants.get('min_vm_number'),constants.get('max_vm_number')),
                          errmsg='Not implemented yet.')

class RequestException(Exception):
    def __init__(self, status_code, reason, *args):
        super(RequestException, self).__init__(*args)
        self.status_code, self.reason = status_code, reason

    def to_dict(self):
        return dict(status_code=self.status_code,
                    reason=self.reason,
                    message=str(self))


@app.errorhandler(RequestException)
def handled_exception(error):
    global logger
    logger.error(f'An exception occured: {error.to_dict()}')
    return jsonify(error.to_dict())


@app.errorhandler(Exception)
def unhandled_exception(error):
    global logger
    import traceback as tb
    logger.error('An unhandled exception occured: %r\n%s',
                 error, tb.format_exc(error))
    response = jsonify(dict(message=error.args))
    response.status_code = 500
    return response

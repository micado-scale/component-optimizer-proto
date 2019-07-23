from flask import Flask, jsonify, request, send_file

from ruamel import yaml

import logging
import logging.config

import opt_config
import opt_utils

import advice
#from advice_oop import Advice
from train import TrainingUnit


app = Flask(__name__)

logger = None
config = None
training_unit = None
#advice = None
training_result = []

constants = {}
sample_number = 0
vm_number_prev = 0
vm_number_prev_kept = 0
min_same_length = 3
sample_data_temp = []

def init_service(cfg):
    global logger
    logger = logging.getLogger('optimizer')

    global config
    config = opt_config.OptimizerConfig(cfg, 'optimizer')




@app.route('/optimizer/init', methods=['POST'])
def init():
    logger.debug('Loading constants from file...')
    constants_yaml = request.stream.read()
    if not constants_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        global constants
        constants = yaml.safe_load(constants_yaml).get('constants')
        logger.debug(f'Constants received: {constants}')

        logger.debug('Saving constants...')
        opt_utils.write_yaml(config.constants_filename, constants)
        logger.debug('Constants saved to "data/constants.yaml"')
        global vm_number_prev
        logger.debug(f'VM_NUMBER_PREV BEFORE INIT: {vm_number_prev}')
        vm_number_prev = constants.get('initial_vm_number', 1)
        global vm_number_prev_kept
        vm_number_prev_kept = vm_number_prev
        logger.debug(f'Initial vm number: {vm_number_prev}')
        logger.debug(f'Vm number prev used: {vm_number_prev_kept}')

        logger.debug('Preparing database for training data...')

        input_metrics = [metric.get('name')
                         for metric in constants.get('input_metrics')]
        target_metrics = [metric.get('name')
                          for metric in constants.get('target_metrics')]


        timestamp_col = ['timestamp']
        vm_cols = ['vm_number', 'vm_number_prev', 'vm_number_diff']

        logger.debug('Creating a .csv file for neural network...')
        opt_utils.persist_data(
            config.nn_filename, timestamp_col+input_metrics+target_metrics, 'w')
        logger.debug('File created')

        logger.debug('Creating a .csv file for linear regression...')
        opt_utils.persist_data(
            config.lr_filename, timestamp_col+input_metrics+vm_cols, 'w')
        logger.debug('File created')

        global training_unit
        global advice

        #hianyzo default value??
        training_unit = TrainingUnit(config, input_metrics, target_metrics, constants.get('target_metrics'), constants.get('max_number_of_scaling_activity', 100), constants.get('nn_stop_error_rate', 10.0), constants.get('max_delta_vm', 2))
        logger.debug('Training unit created.')
        advice.init(constants.get('target_metrics'), constants.get('training_samples_required', 10), constants.get('min_vm_number', 1), constants.get('max_vm_number', 10), constants.get('max_delta_vm', 2), constants.get('nn_stop_error_rate', 10.0))
        #advice = Advice(constants.get('target_metrics'), constants.get('training_samples_required', 10), constants.get('min_vm_number', 1), constants.get('max_vm_number', 10), constants.get('max_delta_vm', 2), constants.get('nn_stop_error_rate', 10.0))
        logger.debug('Advice created.')
        
        logger.info('Optimizer REST initialized successfully ')
    return jsonify('OK'), 200


@app.route('/optimizer/training_data', methods=['GET', 'POST'])
def training_data():
    if request.method == 'GET':
        logger.info('Training_data requested')
        logger.debug(
            'Zipping neural network and linear regression training data files...')
        zip_file = config.training_data_filename
        opt_utils.zip_files([config.nn_filename, config.lr_filename], zip_file)
        logger.debug('Files zipped')
        logger.info('Sending zipped training data...')
        return send_file(zip_file,
                         mimetype='zip',
                         attachment_filename=zip_file,
                         as_attachment=True)


@app.route('/optimizer/sample', methods=['POST'])
def sample():
    logger.debug('Loading training sample...')
    sample_yaml = request.stream.read()
    if not sample_yaml:
        raise RequestException(400, 'Empty POST data')
    else:
        sample = yaml.safe_load(sample_yaml)
        logger.debug(f'New sample received: {sample}')
        logger.debug('Getting sample data...')
        input_metrics = [metric.get('value')
                         for metric in sample.get('sample').get('input_metrics')]
        target_metrics = [metric.get('value')
                          for metric in sample.get('sample').get('target_metrics')]
        vm_number = sample.get('sample').get('vm_number')
        timestamp_col = [sample.get('sample').get('timestamp')]
        logger.debug('Sample data stored in corresponding variables.')

        print(timestamp_col+input_metrics+target_metrics+[vm_number])
        if None not in timestamp_col+input_metrics+target_metrics+[vm_number]: 
            logger.debug('Sample accepted.')
            
            global vm_number_prev
            global vm_number_prev_kept
            global sample_number
            global sample_data_temp
            logger.debug(f'VM number now: {vm_number}')
            logger.debug(f'VM number prev: {vm_number_prev}')

            if vm_number == vm_number_prev:
                sample_data_temp.append([timestamp_col+input_metrics+target_metrics, timestamp_col+input_metrics+[vm_number, vm_number_prev, vm_number-vm_number_prev]])
                logger.debug('Sample data added to temporary buffer.')
            else:
                logger.debug('New VM number occurred.')
                logger.debug(f'Sample data buffer length now: {len(sample_data_temp)}')
                logger.debug(f'Min same length: {min_same_length}')
                if len(sample_data_temp) >= min_same_length:
                    logger.debug('Enough valid data with the same VM number, start persisting...')
                    logger.debug(f'Sample data to persist: {sample_data_temp}')
                    nn_data, lr_data = map(list, zip(*sample_data_temp))
                    logger.debug(f'NN data: {nn_data}')
                    logger.debug(f'LR data: {lr_data}')
                    logger.debug('Saving timestamp, input and target metrics to neural network data file...') 
                    opt_utils.persist_data(config.nn_filename, nn_data, 'a')
                    logger.debug('Data saved')

                    logger.debug('Saving timestamp, input metrics and VM number related data to linear regression data file...')
                    opt_utils.persist_data(config.lr_filename, lr_data, 'a')
                    logger.debug('Data saved')
                    
                    vm_number_prev_kept = vm_number_prev
                    logger.debug(f'Vm number prev used after persisting relevant data: {vm_number_prev_kept}')

                    sample_number += len(sample_data_temp)
                    logger.debug(f'Number of samples now: {sample_number}')

                sample_data_temp.clear()
                logger.debug(f'Sample data after clearing: {sample_data_temp}')
                sample_data_temp.append([timestamp_col+input_metrics+target_metrics, timestamp_col+input_metrics+[vm_number, vm_number_prev_kept, vm_number-vm_number_prev_kept]])
                logger.debug(f'Sample data after new append: {sample_data_temp}')

                
            vm_number_prev = vm_number
            logger.info('Samples received and processed.')   

            #training
            if sample_number >= constants.get('training_samples_required', 10):
                logger.debug('Enough valid sample for start training.')
                global training_result
                training_result = training_unit.train()
                logger.debug(f'Training result:  {training_result}')
            else:
                logger.debug('Not enough valid sample for start training.')
        else:
            logger.error('Sample ignored because it contains empty value.')
    return jsonify('OK'), 200


@app.route('/optimizer/advice', methods=['GET'])
def get_advice():
    logger.debug(f'Sample number when advice was called: {sample_number}')
    logger.debug(f'Training result: {training_result}')
    return advice.get_advice(sample_number, *training_result)

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

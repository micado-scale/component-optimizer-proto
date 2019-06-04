from flask import jsonify
import logging

training_samples_required = None
nn_stop_error_rate = None
min_vm_number = None
max_vm_number = None
target_metric_min = None
target_metric_max = None

logger = logging.getLogger('optimizer')

def init(_target_metric_thresholds, _training_samples_required=10, _min_vm_number=1,_max_vm_number=10,_nn_stop_error_rate=10.0):
    global logger
    global training_samples_required
    training_samples_required = _training_samples_required

    global min_vm_number
    min_vm_number = _min_vm_number

    global max_vm_number
    max_vm_number = _max_vm_number

    global nn_stop_error_rate
    nn_stop_error_rate = _nn_stop_error_rate

    global target_metric_min 
    target_metric_min= _target_metric_thresholds[0].get('min_threshold')

    global target_metric_max
    target_metric_max= _target_metric_thresholds[0].get('max_threshold')
    logger.debug('Advice initialized successfully.')


def advice_msg(valid=False, phase='training', vm_number=0, nn_error_rate=1000, error_msg=None):
  if valid:
    return jsonify(dict(valid=valid, phase=phase,vm_number=vm_number, nn_error_rate=nn_error_rate, error_msg='')), 200
  else:
    return jsonify(dict(valid=valid, phase=phase,vm_number=vm_number, nn_error_rate=nn_error_rate,error_msg=error_msg)), 400

def get_advice(sample_number, actual_vm_number=None, predictions=None, nn_error_rate=None, error_msg=None): #error msg comes from training section
    global logger
    global training_samples_required
    global nn_stop_error_rate
    logger.debug('Checking phase...')
    if sample_number == 0:
        msg = 'There are no training samples yet.'
        logger.error(msg)
        return advice_msg(valid=False, phase='invalid', error_msg=msg)
    elif sample_number < training_samples_required:
        logger.debug('PRETRAINING PHASE')
        if sample_number == 1:
            logger.info('Pretraining phase - 1st call')
            global min_vm_number
            global max_vm_number
            return advice_msg(valid=True, phase='pretraining',
                              vm_number=int((max_vm_number+min_vm_number)/2))
        else:
            logger.info('Collecting samples to get training started...')
            return advice_msg(valid=True, phase='pretraining', vm_number=actual_vm_number)
    else:
        if predictions is not None:
            phase = ''
            global target_metric_min
            global target_metric_max
                
            if nn_error_rate > nn_stop_error_rate:
                logger.debug('TRAINING MODE')
                phase = 'training'
            else:
                logger.debug('PRODUCTION MODE')
                phase = 'production'

            best_prediction = min([(target_metric_max+target_metric_min)/2 - pred for pred in list(predictions.values())])
            indices = [ind for ind, val in enumerate(list(predictions.values())) if val == best_prediction]
            needed_ks = [list(predictions.keys())[ind] for ind in indices]
            vm_number_total = actual_vm_number+min(map(abs, needed_ks))
            logger.debug(f'Advice is: {vm_number_total}, phase: {phase}.')
            return advice_msg(valid=True, phase=phase, vm_number=vm_number_total, nn_error_rate=nn_error_rate)                      
            
        else:
            msg = 'Error occurred: empty predictions received: '
            logger.error(msg+error_msg)
            return advice_msg(valid=False,
                          error_msg=msg+error_msg)



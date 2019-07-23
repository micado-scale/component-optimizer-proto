from flask import jsonify
import logging

training_samples_required = None
nn_stop_error_rate = None
min_vm_number = None
max_vm_number = None
max_delta_vm = None
target_metric_min = None
target_metric_max = None

logger = logging.getLogger('optimizer')

def init(_target_metric_thresholds, _training_samples_required=10, _min_vm_number=1,_max_vm_number=10, _max_delta_vm=2, _nn_stop_error_rate=10.0):

    global training_samples_required
    training_samples_required = _training_samples_required

    global min_vm_number
    min_vm_number = _min_vm_number

    global max_vm_number
    max_vm_number = _max_vm_number

    global max_delta_vm
    max_delta_vm = _max_delta_vm

    global nn_stop_error_rate
    nn_stop_error_rate = _nn_stop_error_rate

    global target_metric_min 
    target_metric_min = _target_metric_thresholds[0].get('min_threshold')

    global target_metric_max
    target_metric_max = _target_metric_thresholds[0].get('max_threshold')
    logger.debug('Advice initialized successfully.')


def advice_msg(valid=False, phase='training', vm_number=0, nn_error_rate=1000, error_msg=None):
  if valid:
    return jsonify(dict(valid=valid, phase=phase,vm_number=vm_number, nn_error_rate=nn_error_rate, error_msg='')), 200
  else:
    return jsonify(dict(valid=valid, phase=phase,vm_number=vm_number, nn_error_rate=nn_error_rate,error_msg=error_msg)), 400

def get_advice(sample_number, actual_vm_number=None, predictions=None, nn_error_rate=None, error_msg=None): #error msg comes from training section

    logger.debug('Checking phase...')
    if sample_number == 0:
        msg = 'There are no training samples yet.'
        logger.error(msg)
        return advice_msg(valid=False, phase='invalid', error_msg=msg)
    elif sample_number < training_samples_required:
        logger.debug('PRETRAINING PHASE')
        if sample_number == 1:
            logger.info('Pretraining phase - 1st call')
            return advice_msg(valid=True, phase='pretraining',
                              vm_number=int((max_vm_number+min_vm_number)/2))
        else:
            logger.info('Collecting samples to get training started...')
            return advice_msg(valid=False, phase='pretraining')
    else:
        if predictions is not None:
            logger.debug(f'Actual VM number: {actual_vm_number}')
            logger.debug(f'Predictions: {predictions}')
            logger.debug(f'NN error rate: {nn_error_rate}')
            logger.debug(f'Error message: {error_msg}')

            vm_number_total = actual_vm_number+get_scaling_decision(predictions)
            logger.debug(f'VM number total: {vm_number_total}')
            
            phase = get_phase(nn_error_rate)
            logger.debug(f'Advice is: {vm_number_total}, phase: {phase}.')
            advice_struct = dict(valid=True, phase=phase, vm_number=vm_number_total, nn_error_rate=nn_error_rate)
            logger.debug(f'Advice struct: {advice_struct}')
            return advice_msg(valid=True, phase=phase, vm_number=vm_number_total, nn_error_rate=nn_error_rate)                      
            
        else:
            msg = 'Error occurred: empty predictions received: '
            logger.error(msg+error_msg)
            return advice_msg(valid=False,
                          error_msg=msg+error_msg)


def get_phase(nn_error_rate):
    if nn_error_rate > nn_stop_error_rate:
        logger.debug('TRAINING MODE')
        return 'training'
    else:
        logger.debug('PRODUCTION MODE')
        #lr-hez is kene kötni
        return 'production'

def get_scaling_decision(predictions):
            logger.debug(f'Target min: {target_metric_min}, target max: {target_metric_max}')
            pred_distances = {k: abs((target_metric_max+target_metric_min)/2 - pred) for k, pred in predictions.items()}
            logger.debug(f'Pred distances: {pred_distances}')

            min_pred_distance = min(list(pred_distances.values()))
            logger.debug(f'Min pred distance: {min_pred_distance}')

            #keressük a legjobb predikciót (amelyik k-ját később visszaadjuk)
            #kivalogatjuk az osszeset, ami minimalis. ha több van, akkor advice = 0, ha 1, akkor ez az 1 lesz az decision
            best_ks = [k for k, pred in pred_distances.items() if pred == min_pred_distance]
            logger.debug(f'Best k-s: {best_ks}')
            scaling_decision = 0 if len(best_ks) != 1 else best_ks[0]
            logger.debug(f'Scaling decision: {scaling_decision}')
            return scaling_decision

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

import numpy as np
import pandas as pd

import csv
import opt_config
from opt_utils import read_data

import logging

class TrainingUnit:
    def __init__(self, conf, input_metrics, target_metrics, target_metrics_thresholds, max_number_of_scaling_activity=100, nn_stop_error_rate=10.0, max_delta_vm=2):
        self.conf = conf
        self.input_metrics = input_metrics
        self.target_metrics = target_metrics
        self.input_metric_number = len(input_metrics)
        self.target_metric_number = len(target_metrics)
        self.target_metrics_thresholds = target_metrics_thresholds
        self.max_number_of_scaling_activity = max_number_of_scaling_activity
        self.nn_stop_error_rate = nn_stop_error_rate
        self.max_delta_vm = max_delta_vm

        self.neural_network_model = self.configure_neural_network()
        self.linear_regression_models = self.configure_linear_regression_models()
        
        self.nn_data = pd.read_csv(self.conf.nn_filename, sep=',', header=0, index_col=0)
        self.lr_data = read_data(self.conf.nn_filename, skip_header=True)
        
        self.lr_required_indices = []
        self.lr_required_data = [] 
        self.ind = 0
        self.required_data_size_prev = 0
        self.training_result_prev = None

        self.logger = logging.getLogger('optimizer')

    def configure_neural_network(self):
        return MLPRegressor(hidden_layer_sizes=(int(self.input_metric_number+self.target_metric_number/2),),
                            activation='logistic',
                            solver='adam',
                            learning_rate='adaptive',
                            learning_rate_init=0.01,
                            alpha=0.01,  
                            max_iter=500, 
                            verbose=True,
                            shuffle=True,  
                            random_state=42)  

    def configure_linear_regression_models(self):
        return [LinearRegression()] * self.input_metric_number

    def read_training_data(self):
        self.nn_data = pd.read_csv(self.conf.nn_filename, sep=',', header=0, index_col=0)
        self.lr_data = read_data(self.conf.lr_filename, skip_header=True)


    #returns the structure that advice needs
    def train(self):
        self.logger.info('Training starts now...')
        self.read_training_data()

        nn_predictions = None
        nn_error_rate = 1000.0 #dummy
        error_msg = None #could be warning or just informing
        #!!! todo refactor
        actual_vm_number = self.lr_data[-1][-3]
        self.logger.debug(f'Actual VM number: {actual_vm_number}')

        #only valid for 1 target!
        target_metric_min = self.target_metrics_thresholds[0].get('min_threshold', 0.4)
        target_metric_max = self.target_metrics_thresholds[0].get('max_threshold', 0.75)
        #split data for nn - todo refactor
        X_nn = self.nn_data[self.nn_data.columns[:-self.target_metric_number]]
        y_nn = self.nn_data[self.nn_data.columns[-self.target_metric_number:]]

        self.logger.debug('Fitting neural network model...')
        self.neural_network_model.fit(X_nn, y_nn)
        nn_prediction = self.neural_network_model.predict(X_nn.iloc[-1:])
        self.logger.info(f'Prediction: {nn_prediction}')
        self.logger.debug(f'Weights: {self.neural_network_model.coefs_}')
        self.logger.debug(f'Bias: {self.neural_network_model.intercepts_}')
        

        if target_metric_min <= nn_prediction <= target_metric_max:
            error_msg = 'Predicted value is inside the specified range.'
            self.logger.error(error_msg) 
        else:
            self.logger.info('Predicted value is out of range, calling LR...')
            lr_predictions = []
            k_values = [] 
            #collect all data where dVM != 0 (scaling activity occurred)
            self.logger.debug('Collecting indices of rows where scaling occurred...')
            self.logger.debug(f'Collecting starts from ind = {self.ind}')
            for i in range(self.ind, len(self.lr_data)):
                if (self.lr_data[i][-1] != 0):
                    self.lr_required_indices.append(i)
                    self.logger.debug(f'New row found with index {i}')
            self.logger.debug(f'Required indices for LR: {self.lr_required_indices}')
            
            self.logger.debug(f'Required row count now: {len(self.lr_required_indices)}')
            self.logger.debug(f'Previous required row count: {self.required_data_size_prev}')
            self.logger.debug(f'Difference between them: {len(self.lr_required_indices) - self.required_data_size_prev}')
            if len(self.lr_required_indices) >= 3:
                if len(self.lr_required_indices) - self.required_data_size_prev != 0: 
                    self.logger.info('Enough data for doing linear regression.')
                    self.logger.debug('Collecting required rows...')

                    required_data_start_ind = len(self.lr_required_data)
                    self.logger.debug(f'Collecting lr data starts from here): {required_data_start_ind}')
                    self.logger.debug(f'Collecting lr data ends here: {len(self.lr_required_indices)-1}')

                    for i in range(required_data_start_ind, len(self.lr_required_indices)-1):
                        self.logger.debug(f'i = {i}')
                        self.logger.debug(f'First half of row: {self.lr_data[self.lr_required_indices[i]][1:self.input_metric_number+2]}')
                        self.logger.debug(f'Other half of row: {self.lr_data[self.lr_required_indices[i+1]][1:self.input_metric_number+1]}')
                        self.lr_required_data.append(self.lr_data[self.lr_required_indices[i]][1:self.input_metric_number+2] + self.lr_data[self.lr_required_indices[i+1]][1:self.input_metric_number+1])
                    
                    self.logger.debug(f'Required data for LR now: {self.lr_required_data}')
                    vm_numbers = [row[self.input_metric_number] for row in self.lr_required_data]

                    for k in range(-self.max_delta_vm, self.max_delta_vm+1):
                        self.logger.debug(f'\nNOW WE HAVE k = {k}')

                        vm_numbers_total = [vm_num + k for vm_num in vm_numbers]
                        self.logger.debug(f'vm_numbers_now: {vm_numbers}')
                        self.logger.debug(f'vm_numbers_total: {vm_numbers_total}')

                        nonzero_vm_count = sum(1 for vm_num_total in vm_numbers_total if vm_num_total > 0)
                        if nonzero_vm_count >= 2:
                            self.logger.info('Enough positive vm count data after adding possible dVM-s to actual VM numbers, preparing for LR...')
                            pred_for_a_single_k = []
                            for i in range(self.input_metric_number):
                                X_lr = []
                                y_lr = []
                                self.logger.debug(f'\nBulding model for {i+1}. metric... ')
                                for j in range(len(self.lr_required_data)):
                                    self.logger.debug(f'{j+1}. row in needed data, vm_numbers_total[j]: {vm_numbers_total[j]} ')
                                    if vm_numbers_total[j] > 0:
                                        X_lr.append([self.lr_required_data[j][i]*vm_numbers[j] / vm_numbers_total[j], self.lr_required_data[j][i]*k / vm_numbers_total[j]])
                                        y_lr.append([self.lr_required_data[j][self.input_metric_number+i+1]])
                                    else:
                                        self.logger.debug('Skipping row')
                                self.logger.debug(f'X_lr for the {i+1}. metric: {X_lr}')
                                self.logger.debug(f'y_lr for the {i+1}. metric: {y_lr}\n')
                                self.logger.debug(f'Fitting {i+1}. model...')
                                self.linear_regression_models[i].fit(X_lr, y_lr)
                                #could use all data for prediction instead of one
                                self.logger.debug(f'weights: {self.linear_regression_models[i].coef_}')
                                self.logger.debug(f'bias: {self.linear_regression_models[i].intercept_}')
                                self.logger.debug(f'X_lr[-1]: { X_lr[-1]}')
                                self.logger.debug(f'prediction: {self.linear_regression_models[i].predict([X_lr[-1]])}')
                                error_lr = sqrt(mean_squared_error(y_lr, self.linear_regression_models[i].predict(X_lr)))
                                self.logger.debug(f'RMS for actual metric: {error_lr}')
                                pred_for_a_single_k.append((self.linear_regression_models[i].predict([X_lr[-1]])).item())
                                self.logger.debug('. . . . . . . . . . . . . . . . . . . . . ')
                            self.logger.debug(f'Pred for a single k: {pred_for_a_single_k}')
                            lr_predictions.append(pred_for_a_single_k)
                            k_values.append(k) 
                            self.logger.debug('___________________________________')

                        else: 
                            self.logger.error('Not enough positive vm count data after adding possible dVM-s to actual VM numbers.')

                    self.logger.debug(f'\nLR predictions: {lr_predictions}')
                    #now go back to nn with the predictions
                    nn_predictions = dict(zip(k_values, self.neural_network_model.predict(lr_predictions))) 
                    self.logger.info(f'\nNN Prediction after LR: {nn_predictions}')
                else: 
                    self.logger.debug(f'No new lr data received. Returning previous training result: {self.training_result_prev}')
                    return self.training_result_prev
            else:
                error_msg = 'Not enough data for doing linear regression.'
                self.logger.error(error_msg)

            self.ind = len(self.lr_data) 
            self.required_data_size_prev = len(self.lr_required_indices)
            self.logger.debug(f'ind after one run: {self.ind}')

        self.logger.info('Training over.')
        training_result = [actual_vm_number, nn_predictions, nn_error_rate, error_msg]
        self.training_result_prev = training_result 
        return training_result

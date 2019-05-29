from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

import numpy as np
import pandas as pd

import csv
from opt_utils import read_data

import logging

class TrainingUnit:
    def __init__(self, input_metrics, target_metrics, target_metrics_thresholds, max_number_of_scaling_activity=100, nn_stop_error_rate=10.0, max_delta_vm=2):
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
        #should use real data file placed in the correct folder!
        self.nn_data = pd.read_csv('test_files/nn_sample.csv', sep=',', header=0, index_col=0)
        self.lr_data = read_data('test_files/lr_sample.csv', skip_header=True)
        
        self.lr_required_indices = [0]
        self.lr_required_data = [] 
        self.last_required_ind = 0

        logger = logging.getLogger('optimizer')

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
        self.nn_data = pd.read_csv('test_files/nn_sample.csv', sep=',', header=0, index_col=0)
        self.lr_data = read_data('test_files/lr_sample.csv', skip_header=True)


    #returns the structure that advice needs
    def train(self):
        print('Training starts now...')
        self.read_training_data()

        nn_predictions = None
        nn_error_rate = 1000.0 #temp 
        error_msg = None #could be warning or just informing
        actual_vm_number = self.lr_data[-1][-3]
        print('Actual VM number: ', actual_vm_number)

        #only valid for 1 target!
        target_metric_min = self.target_metrics_thresholds[0].get('min_threshold', 0.4)
        target_metric_max = self.target_metrics_thresholds[0].get('max_threshold', 0.75)
        #split data for nn - todo refactor
        X_nn = self.nn_data[self.nn_data.columns[:-self.target_metric_number]]
        y_nn = self.nn_data[self.nn_data.columns[-self.target_metric_number:]]

        print('LEARNING STARTS NOW.\n')
        print('Fitting neural network model...')
        self.neural_network_model.fit(X_nn, y_nn)
        nn_prediction = self.neural_network_model.predict(X_nn.iloc[-1:])
        print('Prediction: ', nn_prediction)
        print('Weights: ', self.neural_network_model.coefs_)
        print('Bias: ', self.neural_network_model.intercepts_)
        

        if target_metric_min <= nn_prediction <= target_metric_max:
            error_msg = 'Predicted value is inside the specified range.'
            print(error_msg) 
        else:
            print('Predicted value is out of range, calling LR...')
            lr_predictions = []
            k_values = [] 
            #collect all data where dVM != 0 (scaling activity occured)
            for i in range(self.last_required_ind+1, len(self.lr_data)):
                if (self.lr_data[i][-1] != 0):
                    self.lr_required_indices.append(i)
            
            if len(self.lr_required_indices) >= 3: 
                print('Enough data for doing linear regression.')
                print('Collecting required rows...')
                for i in range(self.last_required_ind, len(self.lr_required_indices)-1):
                    self.lr_required_data.append(self.lr_data[self.lr_required_indices[i]][1:self.input_metric_number+2] + self.lr_data[self.lr_required_indices[i+1]][1:self.input_metric_number+1])

                vm_numbers = [row[self.input_metric_number] for row in self.lr_required_data]

                for k in range(-self.max_delta_vm, self.max_delta_vm+1):
                    print(f'\nNOW WE HAVE k = {k}')

                    vm_numbers_total = [vm_num + k for vm_num in vm_numbers]
                    print('vm_numbers_now: ', vm_numbers)
                    print('vm_numbers_total: ', vm_numbers_total)

                    if len(self.lr_required_data) - vm_numbers_total.count(0) >= 2:
                        print('Enough nonzero data after adding possible dVM-s to actual VM numbers, preparing for LR...')
                        pred_for_a_single_k = []
                        for i in range(self.input_metric_number):
                            X_lr = []
                            y_lr = []
                            print(f'\nBulding model for {i+1}. metric... ')
                            for j in range(len(self.lr_required_data)):
                                print(f'{j+1}. row in needed data, vm_numbers_total[j]: {vm_numbers_total[j]} ')
                                if vm_numbers_total[j] > 0:
                                    X_lr.append([self.lr_required_data[j][i]*vm_numbers[j] / vm_numbers_total[j], self.lr_required_data[j][i]*k / vm_numbers_total[j]])
                                    y_lr.append([self.lr_required_data[j][self.input_metric_number+i+1]])
                                else:
                                    print('Skipping row')
                            print(f'X_lr for the {i+1}. metric: {X_lr}')
                            print(f'y_lr for the {i+1}. metric: {y_lr}\n')
                            print(f'Fitting {i+1}. model...')
                            self.linear_regression_models[i].fit(X_lr, y_lr)
                            #lehetne az összessel prediktálni, mert úgyis kelleni fognak az error kiméréséhez
                            print('weights: ', self.linear_regression_models[i].coef_)
                            print('bias: ', self.linear_regression_models[i].intercept_)
                            print('X_lr[-1]: ', X_lr[-1])
                            print('prediction: ', self.linear_regression_models[i].predict([X_lr[-1]]))
                            error_lr = sqrt(mean_squared_error(y_lr, self.linear_regression_models[i].predict(X_lr)))
                            print('RMS for actual metric: ', error_lr)
                            pred_for_a_single_k.append((self.linear_regression_models[i].predict([X_lr[-1]])).item())
                            print('. . . . . . . . . . . . . . . . . . . . . ')
                            #TODO: print actual metric value
                        print('Pred for a single k: ', pred_for_a_single_k)
                        lr_predictions.append(pred_for_a_single_k)
                        k_values.append(k) 
                        print('___________________________________')

                    else: 
                        print('Not enough nonzero data after adding possible dVM-s to actual VM numbers.')

                print('\nLR predictions: ', lr_predictions)
                #now go back to nn with the predictions
                nn_predictions = dict(zip(k_values, self.neural_network_model.predict(lr_predictions))) 
                print('\nNN Prediction after LR: ', nn_predictions)
            else:
                error_msg = 'Not enough data for doing linear regression.'
                print('Last required row index after one run: ', self.last_required_ind)
            self.last_required_ind = self.lr_required_indices[-1] 

        return [actual_vm_number, nn_predictions, nn_error_rate, error_msg]

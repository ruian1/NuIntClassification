import json

# Create configuration for a grid search on certain gcnn configurations

learning_rates = [1e-3, 5e-3, 1e-4]
hidden_units = [
    [64, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 64, 64]
]
use_bns = [True, False]
use_residuals = [True, False]
lr_strategies = ['reduce_on_plateau', 'exponential_decay']
min_track_lengths = [None, 70]
max_cascade_energies = [None, 10]


for eta in learning_rates:
    for hidden_unit in hidden_units:
        for min_track_length in min_track_lengths:
            for max_cascade_energy in max_cascade_energies:
                for use_bn in use_bns:
                    for use_residual in use_residuals:
                        for lr_strategy in lr_strategies:
                            suffix = '_'.join(map(str, ['-'.join(map(str, hidden_unit)), eta, min_track_length, max_cascade_energy, use_bn, use_residual, lr_strategy]))
                            file_name = f'{suffix}.json'
                            config = {
                                'dataset' : {
                                    'type' : 'hdf5',
                                    'path' : '../data/data_dragon4.hd5',
                                    'shuffle' : True,
                                    'features' : [
                                        "ChargeFirstPulse", "TimeFirstPulse", "ChargeLastPulse", "TimeLastPulse", "TimeVariance", "IntegratedCharge",
                                        "VertexX", "VertexY", "VertexZ"
                                    ],
                                    'distances_precomputed' : True,
                                    'balance_classes' : True,
                                    'min_track_length' : min_track_length,
                                    'max_cascade_energy' : max_cascade_energy,
                                },
                                'model' : {
                                    'type' : 'gcnn',
                                    'hidden_units_graph_convolutions' : hidden_unit,
                                    'hidden_units_fully_connected' : [1],
                                    'use_batchnorm' : use_bn,
                                    'dropout_rate' : 0.5
                                },
                                'training' : {
                                    'learning_rate' : eta,
                                    'directory' : './training/gcnn_grid/hd5_{0}/',
                                    'use_class_weights' : True,
                                    'logfile' : f'./log/gcnn_grid/{file_name}.log',
                                    'epochs' : 25,
                                }
                            }
                            with open(f'settings/gcnn_grid/{file_name}', 'w+') as f:
                                json.dump(config, f, indent='\t')

import json

# Create configuration for a grid search on certain gcnn configurations

learning_rates = [1e-3, 1e-4]
use_batchnorms = [False, True]
hidden_units = [
    [64, 64, 64],
    [64, 64, 64, 64],
    [64, 64, 64, 64, 64],
    [64, 64, 128, 64, 64],
    [64, 64, 64, 64, 64, 64],
    [64, 64, 64, 64, 64, 64, 32],
]
fc_units = [
    [32, 1],
    [32, 32, 1]
]

for eta in learning_rates:
    for use_batchnorm in use_batchnorms:
        for hidden_unit in hidden_units:
            for fc_unit in fc_units:
                file_name = f'{"-".join(map(str, hidden_unit))}_{eta}_{use_batchnorm}_{"-".join(map(str, fc_unit))}.json'
                config = {
                    'dataset' : {
                        'type' : 'hdf5',
                        'path' : '../data/data_dragon4.hd5',
                        'shuffle' : True,
                        'features' : [
                            "ChargeFirstPulse", "TimeFirstPulse", "ChargeLastPulse", "TimeLastPulse", "TimeVariance", "IntegratedCharge",
                            "VertexX", "VertexY", "VertexZ"
                        ],
                        'distances_precomputed' : True
                    },
                    'model' : {
                        'type' : 'gcnn',
                        'hidden_units_graph_convolutions' : hidden_unit,
                        'hidden_units_fully_connected' : fc_unit,
                        'use_batchnorm' : use_batchnorm,
                        'dropout_rate' : 0.3
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

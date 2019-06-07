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
use_weights = [False, True]

for eta in learning_rates:
    for use_batchnorm in use_batchnorms:
        for hidden_unit in hidden_units:
            for use_weight in use_weights:
                file_name = f'{"-".join(map(str, hidden_unit))}_{eta}_{use_batchnorm}_{use_weight}.json'
                config = {
                    'dataset' : {
                        'type' : 'hdf5',
                        'path' : '../data/data_dragon3.hd5',
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
                        'hidden_units_fully_connected' : [32, 1],
                        'use_batchnorm' : use_batchnorm,
                        'dropout_rate' : 0.3
                    },
                    'training' : {
                        'learning_rate' : eta,
                        'directory' : './gcnn_grid/hd5_{0}/',
                        'use_class_weights' : use_weight,
                        'logfile' : f'log/gcnn_grid/{file_name}.log',
                        'epochs' : 25,
                    }
                }
                with open(f'settings/gcnn_grid/{file_name}', 'w+') as f:
                    json.dump(config, f, indent='\t')
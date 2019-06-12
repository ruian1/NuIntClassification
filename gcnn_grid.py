import json

# Create configuration for a grid search on certain gcnn configurations

learning_rates = [1e-3, 1e-4]
hidden_units = [
    [64, 64, 64, 64],
    [64, 64, 64, 64, 64],
    [64, 64, 64, 64, 64]
]
min_track_lengths = [None, 50]
max_cascade_energies = [None, 10]
data_balances = [
    True, False
]
use_recos = [
    True, False
]

for eta in learning_rates:
    for hidden_unit in hidden_units:
        for min_track_length in min_track_lengths:
            for max_cascade_energy in max_cascade_energies:
                for data_balance in data_balances:
                    for use_reco in use_recos:
                        file_name = f'{"-".join(map(str, hidden_unit))}_{eta}_{min_track_length}_{max_cascade_energy}_{data_balance}_{use_reco}.json'
                        config = {
                            'dataset' : {
                                'type' : 'hdf5',
                                'path' : '../data/data_dragon4.hd5',
                                'shuffle' : True,
                                'features' : ([
                                    "ChargeFirstPulse", "TimeFirstPulse", "ChargeLastPulse", "TimeLastPulse", "TimeVariance", "IntegratedCharge",
                                    "VertexX", "VertexY", "VertexZ", "RecoX", "RecoY", "RecoZ", "RecoZenith", "RecoAzimuth"
                                ] if use_reco else [
                                    "ChargeFirstPulse", "TimeFirstPulse", "ChargeLastPulse", "TimeLastPulse", "TimeVariance", "IntegratedCharge",
                                    "VertexX", "VertexY", "VertexZ"
                                ]),
                                'distances_precomputed' : True,
                                'balance_classes' : data_balance,
                                'min_track_length' : min_track_length,
                                'max_cascade_energy' : max_cascade_energy,
                            },
                            'model' : {
                                'type' : 'gcnn',
                                'hidden_units_graph_convolutions' : hidden_unit,
                                'hidden_units_fully_connected' : [32, 1],
                                'use_batchnorm' : False,
                                'dropout_rate' : 0.2
                            },
                            'training' : {
                                'learning_rate' : eta,
                                'directory' : './training/gcnn_grid/hd5_{0}/',
                                'use_class_weights' : not data_balance,
                                'logfile' : f'./log/gcnn_grid/{file_name}.log',
                                'epochs' : 25,
                            }
                        }
                        with open(f'settings/gcnn_grid/{file_name}', 'w+') as f:
                            json.dump(config, f, indent='\t')

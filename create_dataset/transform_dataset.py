#!/usr/bin/env python3

# Module to transform the dataset by shuffling and splitting into training, test, validation

import numpy as np
import h5py
import os
from collections import defaultdict
import tempfile

VALIDATION_PORTION = 0.1
TESTING_PORTION = 0.1


def create_dataset(f, idxs, dir, prefix, column_types):
    """ Creates a dataset for given indices. """
    vertex_offsets = f['NumberVertices']['value'].cumsum() - f['NumberVertices']['value']
    N_vertices_squared = f['NumberVertices']['value']**2
    distances_offsets = N_vertices_squared.cumsum() - N_vertices_squared


    N_events = idxs.shape[0]
    N_vertices = f['NumberVertices']['value'][idxs].sum()
    N_distances = N_vertices_squared[idxs].sum()

    columns = defaultdict(list)

    with h5py.File(os.path.join(dir, f'{prefix}.hd5'), 'w') as out:
        for key in f.keys():
            if key == '__I3Index__': continue            
            if column_types[key] == 'event':
                shape = (N_events,)
                column = 'value'
            elif column_types[key] == 'vertex':
                shape = (N_vertices,)
                column = 'item'
            elif column_types[key] == 'distance':
                shape = (N_distances,)
                column = 'item'
            elif column_types[key] == 'debug':
                shape = (N_events * 3,)
                column = 'item'
            else:
                raise RuntimeError('Unkown column type for key {key}: {column_types[key]}')
            columns[column_types[key]].append(key)
            out.create_dataset(key, shape=shape, dtype=f[key].dtype[column])

        # Assemble an index set
        vertex_idxs = np.zeros((N_vertices), dtype=np.int64)
        offset = 0
        for i, idx in enumerate(idxs):
            print(f'{i} / {len(idxs)}\r', end='\r')
            size = f['NumberVertices'][idx]['value']
            vertex_offset = vertex_offsets[idx]
            vertex_idxs[offset : offset + size] = np.arange(vertex_offset, vertex_offset + size)
            offset += size

        # Type 'Vertex'
        for key in columns['vertex']:
            print(f'\rCopying {key}...', end='\r')
            data = np.array(f[key]['item'])[vertex_idxs]
            out[key][:] = data

        # Type: 'Event'
        print(f'Copying data for column type \'event\'')
        # Events fit into memory
        for key in columns['event']:
            print(f'\rCopying {key}...', end='\r')
            data = np.array(f[key]['value'])[idxs]
            out[key][:] = data

        # Type: 'Debug', coordinates with 3 values
        print(f'Copying data for column type \'debug\'')
        for key in columns['debug']:
            print(f'\rCopying {key}...', end='\r')
            data = np.array(f[key]['item']).reshape((-1, 3))[idxs]
            out[key][:] = data.reshape((-1))

        

if __name__ == '__main__':
    input = './data/out_feb16.hd'
    output = '/data/user/ran/out_split_feb16'

    os.makedirs(output, exist_ok=True)

    with h5py.File(input) as f:
        N_events = f['NumberVertices'].shape[0]
        N_vertices = f['NumberVertices']['value'].sum()
        N_vertices_squared = (f['NumberVertices']['value']**2).sum()

        # Mark each data column
        column_types = dict()
        for key in f.keys():
            #print("key is ", key)
            if key == '__I3Index__': continue
            N_col = f[key].shape[0]
            if N_col == N_events:
                column_types[key] = 'event'
            elif N_col == N_vertices:
                column_types[key] = 'vertex'
            elif N_col == N_vertices_squared:
                column_types[key] = 'distance'
            elif N_col == N_events * 3:
                column_types[key] = 'debug'
            else:
                raise RuntimeError(f'Unkown column type for key {key} with shape {N_col}')

        # Shuffle the idx
        idx = np.arange(N_events)
        np.random.shuffle(idx)
        first_val_idx = int(TESTING_PORTION * N_events)
        first_train_idx = int((TESTING_PORTION + VALIDATION_PORTION) * N_events)
        idx_test = idx[ : first_val_idx]
        idx_val = idx[first_val_idx : first_train_idx]
        idx_train = idx[first_train_idx : ]
        print(f'#Train: {idx_train.shape[0]} -- #Val: {idx_val.shape[0]} -- #Test: {idx_test.shape[0]}')
        
        create_dataset(f, idx_train, output, 'train', column_types)
        create_dataset(f, idx_test, output, 'test', column_types)
        create_dataset(f, idx_val, output, 'val', column_types)




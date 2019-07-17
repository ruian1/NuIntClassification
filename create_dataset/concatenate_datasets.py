import h5py
from glob import glob
import sys

# Use the 'DeltaLLH' column of the dataset to retrieve the number of events per datafile
PER_EVENT_COLUMN = 'DeltaLLH'

if __name__ == '__main__':
    pattern = sys.argv[1]
    outfile = sys.argv[2]
    paths = glob(pattern)
    print(f'Found {len(paths)} files that match the pattern.')
    
    
    dimensions = {}
    dtypes = {}
    # Calculate the dataset total size beforehand
    for idx, path in enumerate(paths):
        with h5py.File(path) as f:
            for key in f.keys():
                if key == '__I3Index__': continue
                if idx == 0:
                    dimensions[key] = 0
                    dtypes[key] = f[key].dtype
                else:
                    assert key in dimensions, f'Dataset {path} contains key {key} which predecessor were missing.'
                    assert dtypes[key] == f[key].dtype, f'Different dtype {dtypes[key]}'
                dimensions[key] += f[key].shape[0]
        print(f'\rScanned file {idx} / {len(paths)}', end='\r')
    
    print(f'Got these final number of rows: {dimensions}')

    offsets = dict((key, 0) for key in dimensions)

    # Create output file
    with h5py.File(outfile, 'w') as outfile:
        # Create a dataset column for storing filenames
        outfile.create_dataset('filepath', (dimensions[PER_EVENT_COLUMN],), dtype=h5py.special_dtype(vlen=bytes))
        for key in dimensions:
            outfile.create_dataset(key, (dimensions[key],), dtype=dtypes[key])
        print(f'Created output file, filling now...')
        for path in paths:
            with h5py.File(path) as src:
                n_events = src[PER_EVENT_COLUMN].shape[0]
                outfile['filepath'][offsets[PER_EVENT_COLUMN] : offsets[PER_EVENT_COLUMN] + size] = bytes(filepath, encoding='ASCII')
                for key in dimensions:
                    print(f'\rCopying {key} from {path}...                             ', end='\r')
                    size = src[key].shape[0]
                    outfile[key][offsets[key] : offsets[key] + size] = src[key]
                    offsets[key] += size


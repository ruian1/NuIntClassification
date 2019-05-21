from icecube import icetray, dataclasses, dataio, phys_services
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import tables
import numpy as np
import pickle

# Parses i3 files in order to create a (huge) hdf5 file that contains all events of all files
with open('dom_positions.pkl', 'rb') as f:
    dom_positions = pickle.load(f)

def normalize_coordinates(coordinates, weights=None, scale=True, copy=False, epsilon=1e-20):
    """ Normalizes the coordinates matrix by centering it using weighs (like charge).
    
    Parameters:
    -----------
    coordinates : ndarray, shape [N, 3]
        The coordinates matrix to be centered.
    weights : ndarray, shape [N] or None
        The weights for each coordinates.
    scale : bool
        If true, the coordinates will all be scaled to have empircal variance of 1.
    copy : bool
        If true, the coordinate matrix will be copied and not overwritten.
    epslion : float
        Additive value when dividing by the standard deviation.
        
    Returns:
    --------
    coordinates : ndarray, shape [N, 3]
        The normalized coordinate matrix with zero mean and unit variance considering weights.
    """
    if copy:
        coordinates = coordinates.copy()
    mean = np.average(coordinates, axis=0, weights=weights).reshape((1, -1))
    coordinates -= mean
    if scale:
        coordinates /= np.sqrt(np.average(coordinates ** 2, axis=0, weights=weights)) + 1e-20
    return coordinates

def get_events_from_frame(frame):
    """ Extracts data (charge and time of first arrival) as well as their positions from a frame.
    
    Parameters:
    -----------
    frame : ?
        The physics frame to extract data from, which is expected to be "applied". 
        I.e. frame = frame['InIcePulses'].apply(frame) should be done before calling this method.
    
    Returns:
    --------
    dom_features : ndarray, shape [N, 2]
        Feature matrix for the doms that were active during this event.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    doms, vertices, omkeys = [], [], []
    for omkey, pulses in frame:
        cumulative_charge, tmin = 0, np.inf
        dom_position = dom_positions[omkey]
        for pulse in pulses:
            cumulative_charge += pulse.charge
            tmin = min(tmin, pulse.time)
        doms.append([cumulative_charge, tmin, pulses[0].charge])
        vertices.append([dom_position[axis] for axis in ('x', 'y', 'z')])
        omkeys.append(omkey)
    return np.array(doms), np.array(vertices), np.array(omkeys)

def get_normalized_data_from_frame(frame, charge_scale=1.0, time_scale=1e-4, append_coordinates_to_features=True):
    """ Creates normalized features from a frame.
    
    Parameters:
    -----------
    frame : ?
        The data frame to extract an event graph with features from.
    charge_scale : float
        The normalization constant for charge.
    time_scale : float
        The normalization constant for time.
    append_coordinates_to_features : bool
        If the normalized coordinates should be appended to the feature matrix.
    
    Returns:
    --------
    dom_features : ndarray, shape [N, D]
        Feature matrix for the doms that were active during this event. 
        The columns are (cumulativecharge, time of first pulse, charge of first pulse[, x, y, z])
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event. All values are in range [0, 1]
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    
    features, coordinates, omkeys = get_events_from_frame(frame)
    coordinates = normalize_coordinates(coordinates, features[:, 0])
    features[:, 0] *= charge_scale
    features[:, 2] *= charge_scale
    features[:, 1] -= features[:, 1].min()
    features[:, 1] *= time_scale
    if append_coordinates_to_features:
        features = np.hstack((features, coordinates))
    return features, coordinates, omkeys

def process_frame(frame):
    """ Callback for processing a frame and adding relevant data to the hdf5 file. 
    
    Parameters:
    -----------
    frame : ?
        The frame to process.
    """
    pulses = frame['InIcePulses'].apply(frame)
    features, coordinates, _ = get_normalized_data_from_frame(pulses)
    frame['NumberVertices'] = dataclasses.I3Double(features.shape[0])
    frame['CumulativeCharge'] = dataclasses.I3VectorFloat(features[:, 0])
    frame['Time'] = dataclasses.I3VectorFloat(features[:, 1])
    frame['FirstCharge'] = dataclasses.I3VectorFloat(features[:, 2])
    frame['VertexX'] = dataclasses.I3VectorFloat(coordinates[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(coordinates[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(coordinates[:, 2])
    return True

def create_dataset(outfile, infiles):
    """
    Creates a dataset in hdf5 format.

    Parameters:
    -----------
    outfile : str
        Path to the hdf5 file.
    paths : dict
        A list of intput i3 files.
    """
    # TODO process all files
    infiles = infiles
    tray = I3Tray()
    tray.AddModule('I3Reader',
                FilenameList = infiles)
    tray.AddModule(process_frame, 'process_frame')
    tray.AddModule(I3TableWriter, 'I3TableWriter', keys=['NumberVertices', 'CumulativeCharge', 'Time', 'FirstCharge', 'VertexX', 'VertexY', 'VertexZ'], 
                TableService=I3HDFTableService(outfile),
                SubEventStreams=['InIceSplit'],
                BookEverything=False
                )
    # TODO process all not only 10 frames
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    for interaction_type in ('nue', 'numu', 'nutau'):
        print('##### Creating dataset for interaciton type', interaction_type)
        outfile = '/project/rpp-kenclark/fuchsgru/data_dragon_3y_{0}.hd5'.format(interaction_type)
        paths = glob('/project/6008051/hignight/dragon_3y/{0}/*'.format(interaction_type))
        create_dataset(outfile, paths)



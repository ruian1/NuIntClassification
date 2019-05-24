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
with open('/project/6008051/fuchsgru/NuIntClassification/dom_positions.pkl', 'rb') as f:
    dom_positions = pickle.load(f)

def normalize_coordinates(coordinates, weights=None, scale=True, copy=False):
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
        
    Returns:
    --------
    coordinates : ndarray, shape [N, 3]
        The normalized coordinate matrix.
    mean : ndarray, shape [3]
        The means along all axis
    std : ndarray, shape [3]
        The standard deviations along all axis
    """
    if copy:
        coordinates = coordinates.copy()
    mean = np.average(coordinates, axis=0, weights=weights).reshape((1, -1))
    coordinates -= mean
    if scale:
        std = np.sqrt(np.average(coordinates ** 2, axis=0, weights=weights)) + 1e-20
        coordinates /= std
    return coordinates, mean, std
    
def get_events_from_frame(frame, charge_threshold=0.5):
    """ Extracts data (charge and time of first arrival) as well as their positions from a frame.
    
    Parameters:
    -----------
    frame : ?
        The physics frame to extract data from.
    charge_threshold : float
        The threshold for a charge to be considered a valid pulse for the time of the first pulse arrival.
    
    Returns:
    --------
    dom_features : ndarray, shape [N, 8]
        Feature matrix for the doms that were active during this event.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    x = frame['SRTTWOfflinePulsesDC']
    hits = x.apply(frame)
    
    doms, vertices, omkeys = [], [], []
    for omkey, pulses in hits:
        cumulative_charge, tmin, first_charge = 0, np.inf, None
        dom_position = dom_positions[omkey]
        for pulse in pulses:
            cumulative_charge += pulse.charge
            if pulse.charge >= charge_threshold:
                tmin = min(tmin, pulse.time)
                if first_charge is None:
                    first_charge = pulse.charge
        # Features:
        # Cumulative Charge per DOM, Time of first charge above threshold, First charge above threshold,
        # Reconstruction x, Reconstruction y, Reconstrucion z, Reconstruction azimuth, Reconstruction zenith
        doms.append([
            cumulative_charge, tmin, pulses[0].charge, track_reco.pos.x, track_reco.pos.y, 
            track_reco.pos.z, track_reco.dir.azimuth, track_reco.dir.zenith
        ])
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
    dom_features : ndarray, shape [N, 11]
        Feature matrix for the doms that were active during this event.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event. All values are in range [0, 1]
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    
    features, coordinates, omkeys = get_events_from_frame(frame)
    coordinates, coordinate_mean, coordinate_std = normalize_coordinates(coordinates, features[:, 0])
    features[:, 0] *= charge_scale
    features[:, 2] *= charge_scale
    features[:, 1] -= features[:, 1].min()
    features[:, 1] *= time_scale
    # Scale the origin of the track reconstruction to share the same coordinate system
    features[:, 3:6] -= coordinate_mean
    features[:, 3:6] /= coordinate_std
    
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
    # Obtain the PDG Encoding for ground truth
    frame['PDGEncoding'] = dataclasses.I3Double(
        dataclasses.get_most_energetic_neutrino(frame['I3MCTree']).pdg_encoding)
    frame['InteractionType'] = dataclasses.I3Double(frame['I3MCWeightDict']['InteractionType'])
    frame['NumberChannels'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['NchCleaned'])
    frame['TotalCharge'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['DCFiducialPE'])
    features, coordinates, _ = get_normalized_data_from_frame(frame)
    frame['NumberVertices'] = dataclasses.I3Double(features.shape[0])
    frame['CumulativeCharge'] = dataclasses.I3VectorFloat(features[:, 0])
    frame['Time'] = dataclasses.I3VectorFloat(features[:, 1])
    frame['FirstCharge'] = dataclasses.I3VectorFloat(features[:, 2])
    frame['VertexX'] = dataclasses.I3VectorFloat(coordinates[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(coordinates[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(coordinates[:, 2])
    frame['RecoX'] = dataclasses.I3VectorFloat(features[:, 3])
    frame['RecoY'] = dataclasses.I3VectorFloat(features[:, 4])
    frame['RecoZ'] = dataclasses.I3VectorFloat(features[:, 5])
    frame['RecoAzimuth'] = dataclasses.I3VectorFloat(features[:, 6])
    frame['RecoZenith'] = dataclasses.I3VectorFloat(features[:, 7])
    frame['DeltaLLH'] = dataclasses.I3Double(frame['IC86_Dunkman_L6']['delta_LLH']) # Used for a baseline classifcation
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
    infiles = infiles
    tray = I3Tray()
    tray.AddModule('I3Reader',
                FilenameList = infiles)
    tray.AddModule(process_frame, 'process_frame')
    tray.AddModule(I3TableWriter, 'I3TableWriter', keys=[
        'NumberVertices', 'CumulativeCharge', 'Time', 'FirstCharge', 'VertexX', 'VertexY', 'VertexZ',
        'RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith', 'DeltaLLH', 'PDGEncoding', 'IC86_Dunkman_L6_SANTA_DirectCharge',
        'InteractionType', 'NumberChannels', 'TotalCharge'], 
                TableService=I3HDFTableService(outfile),
                SubEventStreams=['InIceSplit'],
                BookEverything=False
                )
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    paths = []
    for interaction_type in ('nue', 'numu', 'nutau'):
        paths += glob('/project/6008051/hignight/dragon_3y/{0}/*'.format(interaction_type))
    outfile = '/project/6008051/fuchsgru/data/data_dragon.hd5'
    create_dataset(outfile, paths)

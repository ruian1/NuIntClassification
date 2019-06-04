from icecube import icetray, dataclasses, dataio, phys_services
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import tables
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances

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
    
def get_events_from_frame(frame, charge_threshold=0.5, time_scale=1e-3, charge_scale=1.0):
    """ Extracts data (charge and time of first arrival) as well as their positions from a frame.
    
    Parameters:
    -----------
    frame : ?
        The physics frame to extract data from.
    charge_threshold : float
        The threshold for a charge to be considered a valid pulse for the time of the first pulse arrival (before scaling).
    charge_scale : float
        The normalization constant for charge.
    time_scale : float
        The normalization constant for time.
    
    Returns:
    --------
    dom_features : ndarray, shape [N, 14]
        Feature matrix for the doms that were active during this event.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    delta_llh = frame['IC86_Dunkman_L6']['delta_LLH']
    x = frame['SRTTWOfflinePulsesDC']
    hits = x.apply(frame)
    
    doms, vertices, omkeys = [], [], []
    for omkey, pulses in hits:
        dom_position = dom_positions[omkey]
        charges, times = [], []
        for pulse in pulses:
            if pulse.charge >= charge_threshold:
                charges.append(pulse.charge)
                times.append(pulse.time)
        times = np.array(times) * time_scale
        charges = np.array(charges) * charge_scale
        if charges.shape[0] == 0: continue # Don't use DOMs that recorded no charge above the threshold
        times -= np.average(times, weights=charges)
        time_std = np.average((times)**2, weights=charges)

        # Features:
        # Charge of first pulse, time of first pulse relative to charge weighted time mean,
        # Charge of last pulse, time of last pulse relative to charge weighted time mean,
        # Charge of the largest pulse, Time of the largest pulse relative to the charge weighted mean,
        # Integrated charge, variance of pulse times from charge weighted time mean,
        # Reconstruction x, Reconstruction y, Reconstrucion z, Reconstruction azimuth, Reconstruction zenith,
        # Delta Log Likelihood (baseline)
        doms.append([
            charges[0], times[0], charges[-1], times[-1], charges.max(), times[charges.argmax()], charges.sum(), time_std,
            track_reco.pos.x, track_reco.pos.y, track_reco.pos.z, track_reco.dir.azimuth, track_reco.dir.zenith, delta_llh
        ])
        vertices.append([dom_position[axis] for axis in ('x', 'y', 'z')])
        omkeys.append(omkey)
        
    return np.array(doms), np.array(vertices), np.array(omkeys)

def get_normalized_data_from_frame(frame, charge_scale=1.0, time_scale=1e-3, append_coordinates_to_features=False):
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
    dom_features : ndarray, shape [N, 14 or 17]
        Feature matrix for the doms that were active during this event.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event. All values are in range [0, 1]
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    features, coordinates, omkeys = get_events_from_frame(frame, charge_scale=charge_scale, time_scale=time_scale)
    coordinates, coordinate_mean, coordinate_std = normalize_coordinates(coordinates, features[:, 0])
    # Scale the origin of the track reconstruction to share the same coordinate system
    features[:, 8:11] -= coordinate_mean
    features[:, 8:11] /= coordinate_std
    
    if append_coordinates_to_features:
        features = np.hstack((features, coordinates))
    return features, coordinates, omkeys

event_offset = 0L
distances_offset = 0L

def process_frame(frame):
    """ Callback for processing a frame and adding relevant data to the hdf5 file. 
    
    Parameters:
    -----------
    frame : ?
        The frame to process.
    """
    global event_offset
    global distances_offset
    # Obtain the PDG Encoding for ground truth
    frame['PDGEncoding'] = dataclasses.I3Double(
        dataclasses.get_most_energetic_neutrino(frame['I3MCTree']).pdg_encoding)
    frame['InteractionType'] = dataclasses.I3Double(frame['I3MCWeightDict']['InteractionType'])
    frame['NumberChannels'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['NchCleaned'])
    frame['TotalCharge'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['DCFiducialPE'])
    features, coordinates, _ = get_normalized_data_from_frame(frame)
    # Calculate pairwise distances
    distances = pairwise_distances(coordinates).reshape([features.shape[0] ** 2])
    frame['NumberVertices'] = icetray.I3Int(features.shape[0])
    frame['Offset'] = icetray.I3Int(event_offset)
    frame['DistancesOffset'] = icetray.I3Int(distances_offset)
    frame['Distances'] = dataclasses.I3VectorFloat(distances)
    event_offset += features.shape[0]
    distances_offset += features.shape[0] ** 2
    frame['ChargeFirstPulse'] = dataclasses.I3VectorFloat(features[:, 0])
    frame['TimeFirstPulse'] = dataclasses.I3VectorFloat(features[:, 1])
    frame['ChargeLastPulse'] = dataclasses.I3VectorFloat(features[:, 2])
    frame['TimeLastPulse'] = dataclasses.I3VectorFloat(features[:, 3])
    frame['ChargeMaxPulse'] = dataclasses.I3VectorFloat(features[:, 4])
    frame['TimeMaxPulse'] = dataclasses.I3VectorFloat(features[:, 5])
    frame['IntegratedCharge'] = dataclasses.I3VectorFloat(features[:, 6])
    frame['TimeVariance'] = dataclasses.I3VectorFloat(features[:, 7])
    frame['RecoX'] = dataclasses.I3VectorFloat(features[:, 8])
    frame['RecoY'] = dataclasses.I3VectorFloat(features[:, 9])
    frame['RecoZ'] = dataclasses.I3VectorFloat(features[:, 10])
    frame['RecoAzimuth'] = dataclasses.I3VectorFloat(features[:, 11])
    frame['RecoZenith'] = dataclasses.I3VectorFloat(features[:, 12])
    frame['VertexDeltaLLH'] = dataclasses.I3VectorFloat(features[:, 13])
    frame['VertexX'] = dataclasses.I3VectorFloat(coordinates[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(coordinates[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(coordinates[:, 2])
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
    global event_offset
    global distances_offset
    event_offset, distances_offset = 0L, 0L
    infiles = infiles
    tray = I3Tray()
    tray.AddModule('I3Reader',
                FilenameList = infiles)
    tray.AddModule(process_frame, 'process_frame')
    tray.AddModule(I3TableWriter, 'I3TableWriter', keys=[
        'NumberVertices', 'ChargeFirstPulse', 'TimeFirstPulse', 'ChargeLastPulse', 'TimeLastPulse', 
        'ChargeMaxPulse', 'TimeMaxPulse', 'IntegratedCharge', 'TimeVariance', # Standard pulse attributes
        'RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith', 'VertexDeltaLLH', # Reconstruction attributes
        'Distances', 'DistancesOffset', # Precomputed distance matrices
        'VertexX', 'VertexY', 'VertexZ', # Coordinates
        'DeltaLLH', 'PDGEncoding', 'IC86_Dunkman_L6_SANTA_DirectCharge', # Metadata
        'InteractionType', 'NumberChannels', 
        'TotalCharge', 'Offset' # Offsets to access flattened data
        ], 
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
    outfile = '/project/6008051/fuchsgru/data/data_dragon3.hd5'
    create_dataset(outfile, paths)

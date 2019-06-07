from icecube import icetray, dataclasses, dataio, phys_services
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import tables
import numpy as np
import sys
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
    
def get_events_from_frame(frame, charge_threshold=0.5, time_scale=1e-3, charge_scale=1.0, max_num_steps=64):
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
    max_num_steps : int
        The maximum sequence length (events with more pulses will be aggregated), events with less pulses zero padded
    
    Returns:
    --------
    event_features : ndarray, shape [num_steps, N, 8]
        Feature matrix for the doms that were active during this event.
    active_vertex : ndarray, shape [num_steps]
        Index of the vertex with greatest idx that is active during each time step.
        The order of the vertices ensure that vertices with smaller idx are active as well.
        Thus, this can be used to easily create the adjacency matrix mask for each graph.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [num_steps, N, 3]
        Omkeys for the doms that were active during this event.
    """
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    x = frame['SRTTWOfflinePulsesDC']
    hits = x.apply(frame)
    
    # Create an evolution of the graph
    events, event_to_vertex, vertices_unordered, omkeys = [], [], [], []
    for vertex_idx, (omkey, pulses) in enumerate(hits):
        dom_position = dom_positions[omkey]
        vertices_unordered.append([dom_position[axis] for axis in ('x', 'y', 'z')])
        cumulative_charge = 0
        for pulse in pulses:
            cumulative_charge += pulse.charge
            if pulse.charge >= charge_threshold:
                events.append(np.array([
                    pulse.charge, cumulative_charge, pulse.time, 
                    0, # placeholder for the time of the previous event which is calculated after sorting
                    track_reco.pos.x, track_reco.pos.y, 
                    track_reco.pos.z, track_reco.dir.azimuth, track_reco.dir.zenith,
                    vertex_idx
                ]))
                time_previous_event = pulse.time
                event_to_vertex.append(vertex_idx)
        assert omkey not in omkeys
        omkeys.append(omkey)

    # Sort events by their time
    events.sort(key = lambda event: event[2])
    events = np.array(events)

    # Sort vertices by their first arrival time
    # This ensures that if a vertex becomes active then all vertices with a lower idx will be active as well
    # which makes the mask for the adjacency matrix a block matrix like: 
    # [1s, 0s]
    # [0s, 0s]
    vertices, vertex_idxs = [], {}
    for event in events:
        vertex_idx = int(event[-1])
        # Any vertex that is newly active will be inserted into the vertex array
        # and assigned a new idx (= position in the vertex array)
        if vertex_idx not in vertex_idxs:
            vertex_idxs[vertex_idx] = len(vertices)
            vertices.append(vertices_unordered[vertex_idx])
        # Reassign the vertex idx (= position in the vertex array)
        event[-1] = vertex_idxs[vertex_idx]
    assert len(vertices) <= len(vertices_unordered)

    events[1 :, 3] = events[1 :, 2] - events[ : -1, 2] # Time difference of events
    events[:, 0:2] *= charge_scale
    events[:, 2:4] *= time_scale
    # Compute charge weighted mean time
    mean_time = np.average(events[:, 2], weights = events[:, 0])
    events[:, 2:4] -= mean_time

    num_steps = min(max_num_steps, len(events))
    event_features = np.zeros((num_steps, len(vertices), events.shape[1] - 1))
    active_vertex = np.zeros(num_steps, dtype=np.int)
    if len(events) <= max_num_steps:
        for idx, event in enumerate(events):
            # From the time step of the arrival of the event on it will be visible in following events
            event_features[idx : , int(event[-1]), : ] = event[ : -1]
            active_vertex[idx] = int(event[-1]) # The reordering ensured that all vertices of events before this one have a lower index 
    else:
        for idx in range(max_num_steps):
            idx_from, idx_to = int(idx * len(events) / max_num_steps), int((idx + 1) * len(events) / max_num_steps)
            # Aggregate events as one time step
            for event_idx in range(idx_from, idx_to):
                event_features[idx :, int(events[event_idx, -1]), :] = events[event_idx, : -1]
                active_vertex[idx] = int(events[event_idx, -1]) # The reordering ensured that all vertices of events before this one have a lower index
        
    return event_features, active_vertex, np.array(vertices), np.array(omkeys)

def get_normalized_data_from_frame(frame, charge_scale=1.0, time_scale=1e-3, max_num_steps=64, append_coordinates_to_features=True):
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
    max_num_steps : int
        The maximum sequence length (events with more pulses will be aggregated), events with less pulses zero padded
        
    Returns:
    --------
    dom_features : ndarray, shape [num_steps, N, 9 or 12]
        Feature matrix for the doms that were active during this event.
    active_vertex : ndarray, shape [num_steps]
        Index of the vertex with greatest idx that is active during each time step.
        The order of the vertices ensure that vertices with smaller idx are active as well.
        Thus, this can be used to easily create the adjacency matrix mask for each graph.
    dom_positions : ndarray, shape [N, 3]
        Coordinate matrix for the doms that were active during this event. All values are in range [0, 1]
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    features, active_vertex, coordinates, omkeys = get_events_from_frame(frame, charge_scale=charge_scale, time_scale=time_scale, max_num_steps=max_num_steps)
    coordinates, coordinate_mean, coordinate_std = normalize_coordinates(coordinates, None)
    # Scale the origin of the track reconstruction to share the same coordinate system
    features[:, :, 4:7] -= coordinate_mean
    features[:, :, 4:7] /= coordinate_std
    
    if append_coordinates_to_features:
        num_steps = features.shape[0]
        #features = np.stack((features, np.repeat(features[np.newaxis, :, :], num_steps)))
        features = np.concatenate((features, np.repeat(coordinates[np.newaxis, :, :], num_steps, axis=0)), axis=-1)

    return features, active_vertex, coordinates, omkeys

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
    features, active_vertex, coordinates, _ = get_normalized_data_from_frame(frame)
    frame['ActiveVertex'] = dataclasses.I3VectorInt(active_vertex.astype(np.int))
    # Compute pairwise distances
    distances = pairwise_distances(coordinates).reshape(-1)
    frame['Distances'] = dataclasses.I3VectorFloat(distances)
    num_steps, num_vertices, _ = features.shape
    frame['NumberVertices'] = icetray.I3Int(num_vertices)
    frame['NumberSteps'] = icetray.I3Int(num_steps)
    # Concatenate the feature columns for an event over all its time steps
    features = features.reshape((num_steps * num_vertices, -1))
    frame['Charge'] = dataclasses.I3VectorFloat(features[:, 0])
    frame['CumulativeCharge'] = dataclasses.I3VectorFloat(features[:, 1])
    frame['Time'] = dataclasses.I3VectorFloat(features[:, 2])
    frame['RelativeTime'] = dataclasses.I3VectorFloat(features[:, 3])
    frame['RecoX'] = dataclasses.I3VectorFloat(features[:, 4])
    frame['RecoY'] = dataclasses.I3VectorFloat(features[:, 5])
    frame['RecoZ'] = dataclasses.I3VectorFloat(features[:, 6])
    frame['RecoAzimuth'] = dataclasses.I3VectorFloat(features[:, 7])
    frame['RecoZenith'] = dataclasses.I3VectorFloat(features[:, 8])
    # Redundant saving of vertex coordinates to match the shape and enhance computational effort when learning
    frame['PulseX'] = dataclasses.I3VectorFloat(features[:, 9])
    frame['PulseY'] = dataclasses.I3VectorFloat(features[:, 10])
    frame['PulseZ'] = dataclasses.I3VectorFloat(features[:, 11])
    frame['VertexX'] = dataclasses.I3VectorFloat(coordinates[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(coordinates[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(coordinates[:, 2])
    frame['DeltaLLH'] = dataclasses.I3Double(frame['IC86_Dunkman_L6']['delta_LLH']) # Used for a baseline classifcation
    frame['NeutrinoEnergy'] = dataclasses.I3Double(frame['trueNeutrino'].energy)
    frame['CascadeEnergy'] = dataclasses.I3Double(frame['trueCascade'].energy)
    try:
        # Appearently also frames with no primary muon contain this field, so to distinguish try to access it (which should throw an exception)
        frame['MuonEnergy'] = dataclasses.I3Double(frame['trueMuon'].energy)
        frame['TrackLength'] = dataclasses.I3Double(frame['trueMuon'].length)
    except:
        frame['MuonEnergy'] = dataclasses.I3Double(np.nan)
        frame['TrackLength'] = dataclasses.I3Double(np.nan)
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
        'NumberVertices', 'NumberSteps', 'Distances', 'ActiveVertex', # Offset data to rebuild matrices from flattened vectors
        'Charge', 'CumulativeCharge', 'Time', 'RelativeTime', # Base features
        'RecoX', 'RecoY', 'RecoZ', 'RecoAzimuth', 'RecoZenith', # Reco features
        'PulseX', 'PulseY', 'PulseZ', # Spatial features (same shape as features, i.e. repeated over time steps)
        'VertexX', 'VertexY', 'VertexZ', # Coordinates (not repeated over time steps)
        'DeltaLLH', 'PDGEncoding', 'IC86_Dunkman_L6_SANTA_DirectCharge', # Meta
        'NeutrinoEneregy', 'CascadeEnergy', 'MuonEnergy', 'TrackLength',
        'InteractionType', 'NumberChannels', 'TotalCharge', ], 
                TableService=I3HDFTableService(outfile),
                SubEventStreams=['InIceSplit'],
                BookEverything=False
                )
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    file_idx = int(sys.argv[1])
    paths = []
    for interaction_type in ('nue', 'numu', 'nutau'):
        paths += glob('/project/6008051/hignight/dragon_3y/{0}/*'.format(interaction_type))
    assert(len(paths) == 1104)
    print('Processing file ' + str(file_idx))
    outfile = '/project/6008051/fuchsgru/data/data_dragon_sequential_parts/{0}.hd5'.format(file_idx)
    create_dataset(outfile, [paths[file_idx]])


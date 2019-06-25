from icecube import icetray, dataclasses, dataio, phys_services
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import sys
import tables
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances
from icecube import NuFlux # genie_icetray

# Parses i3 files in order to create a (huge) hdf5 file that contains all events of all files
with open('/project/6008051/fuchsgru/NuIntClassification/dom_positions.pkl', 'rb') as f:
    dom_positions = pickle.load(f)

flux_service  = NuFlux.makeFlux("IPhonda2014_spl_solmin")

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
        std = np.sqrt(np.average(coordinates ** 2, axis=0, weights=None)) + 1e-20
        coordinates /= std
    return coordinates, mean.flatten(), std

vertex_features = [
    'ChargeFirstPulse', 'ChargeLastPulse', 'ChargeMaxPulse', 'TimeFirstPulse', 'TimeLastPulse', 'TimeMaxPulse',
    'TotalCharge', 'TimeStd',  'TimeFirstPulseShifted', 'TimeLastPulseShifted', 'TimeMaxPulseShifted',
    'TotalChargeShifted', 'TimeStdShifted', 

]

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
    dom_features : dict
        A dict from features to ndarrays of shape [N,] representing the features for each vertex.
    vertices : dict
        A dict from coordiante axis to ndarrays of shape [N,] representing the positions for each vertex on an axis.
        Coordinate matrix for the doms that were active during this event.
    omkeys : ndarray, shape [N, 3]
        Omkeys for the doms that were active during this event.
    """
    x = frame['SRTTWOfflinePulsesDC']
    hits = x.apply(frame)
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    track_reco.shape = dataclasses.I3Particle.InfiniteTrack

    features = {feature : [] for feature in vertex_features}
    vertices = {axis : [] for axis in ('x', 'y', 'z')}
    
    # First compute the charge weighted average time of the entire event
    charges, times = [], []
    for omkey, pulses in hits:
        for pulse in pulses:
            if pulse.charge >= charge_threshold:
                charges.append(pulse.charge)
                times.append(pulse.time)
    times = np.array(times) * time_scale
    average_time = np.average(times, weights=charges)

    # For each event compute event features based on the pulses
    omkeys = []
    for omkey, pulses in hits:
        dom_position = dom_positions[omkey]
        charges, times, times_shifted = [], [], []
        # Calculate the expected time of the charge at the DOM assuming a correct track reconstruction
        cherenkov_time = phys_services.I3Calculator.cherenkov_time(
            frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track'],
            dataclasses.I3Position(dom_position['x'], dom_position['y'], dom_position['z']))
        expected_time = cherenkov_time + track_reco.time
        for pulse in pulses:
            if pulse.charge >= charge_threshold:
                charges.append(pulse.charge)
                times.append(pulse.time)
                times_shifted.append(pulse.time - expected_time)
        times = np.array(times) * time_scale
        times_shifted = np.array(times_shifted) * time_scale
        charges = np.array(charges) * charge_scale
        if charges.shape[0] == 0: continue # Don't use DOMs that recorded no charge above the threshold
        times -= average_time
        time_std = np.sqrt(np.average((times)**2, weights=charges))
        time_std_shifted = np.sqrt(np.average((times_shifted)**2, weights=charges))
        features['ChargeFirstPulse'].append(charges[0])
        features['ChargeLastPulse'].append(charges[-1])
        features['ChargeMaxPulse'].append(charges.max())
        features['TimeFirstPulse'].append(times[0])
        features['TimeLastPulse'].append(times[-1])
        features['TimeMaxPulse'].append(times[charges.argmax()])
        features['TimeStd'].append(time_std)
        features['TimeFirstPulseShifted'].append(times_shifted[0])
        features['TimeLastPulseShifted'].append(times_shifted[-1])
        features['TimeMaxPulseShifted'].append(times_shifted[charges.argmax()])
        features['TimeStdShifted'].append(time_std_shifted)
        features['TotalCharge'].append(charges.sum())

        for axis in vertices:
            vertices[axis].append(dom_position[axis])
        assert omkey not in omkeys
        omkeys.append(omkey)
        #print(omkeys)
        
    return features, vertices, np.array(omkeys)


def get_weight_by_flux(frame):
    """ Calculates the weight of an event using the fluxes.
    
    Parameters:
    -----------
    frame : ?
        The frame to process.
    """
    true_neutrino = dataclasses.get_most_energetic_neutrino(frame["I3MCTree"])
    true_nu_energy  = true_neutrino.energy
    true_nu_coszen  = np.cos(true_neutrino.dir.zenith)
    norm = (frame["I3MCWeightDict"]['OneWeight'] / frame["I3MCWeightDict"]['NEvents']) * 2.
    if (true_neutrino.type > 0):
        nue_flux  = flux_service.getFlux(dataclasses.I3Particle.NuE , true_nu_energy, true_nu_coszen) * norm*0.5/0.7
        numu_flux = flux_service.getFlux(dataclasses.I3Particle.NuMu, true_nu_energy, true_nu_coszen) * norm*0.5/0.7
    else:
        nue_flux  = flux_service.getFlux(dataclasses.I3Particle.NuEBar , true_nu_energy, true_nu_coszen) * norm*0.5/0.3
        numu_flux = flux_service.getFlux(dataclasses.I3Particle.NuMuBar, true_nu_energy, true_nu_coszen) * norm*0.5/0.3
    frame['NuMuFlux'] = dataclasses.I3Double(numu_flux)
    frame['NueFlux'] = dataclasses.I3Double(nue_flux)
    frame['NoFlux'] = dataclasses.I3Double(norm)
    return True


event_offset = 0L
distances_offset = 0L

def process_frame(frame, charge_scale=1.0, time_scale=1e-3, append_coordinates_to_features=False):
    """ Processes a frame to create an event graph and metadata out of it.
    
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
    """
    global event_offset
    global distances_offset

    ### Meta data of the event for analysis of the classifier and creation of ground truth
    nu = dataclasses.get_most_energetic_neutrino(frame['I3MCTree'])

    # Obtain the PDG Encoding for ground truth
    frame['PDGEncoding'] = dataclasses.I3Double(nu.pdg_encoding)
    frame['InteractionType'] = dataclasses.I3Double(frame['I3MCWeightDict']['InteractionType'])
    frame['NumberChannels'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['NchCleaned'])
    frame['DCFiducialPE'] = dataclasses.I3Double(frame['IC86_Dunkman_L3_Vars']['DCFiducialPE'])
    frame['NeutrinoEnergy'] = dataclasses.I3Double(frame['trueNeutrino'].energy)
    # Some rare events do not produce a cascade
    try:
        frame['CascadeEnergy'] = dataclasses.I3Double(frame['trueCascade'].energy)
    else:
        frame['CascadeEnergy'] = dataclasses.I3Double(np.nan)
    try:
        # Appearently also frames with no primary muon contain this field, so to distinguish try to access it (which should throw an exception)
        frame['MuonEnergy'] = dataclasses.I3Double(frame['trueMuon'].energy)
        frame['TrackLength'] = dataclasses.I3Double(frame['trueMuon'].length)
    except:
        frame['MuonEnergy'] = dataclasses.I3Double(np.nan)
        frame['TrackLength'] = dataclasses.I3Double(np.nan)
    frame['DeltaLLH'] = dataclasses.I3Double(frame['IC86_Dunkman_L6']['delta_LLH']) # Used for a baseline classifcation

    ### Create features for each event 
    features, coordinates, _ = get_events_from_frame(frame, charge_scale=charge_scale, time_scale=time_scale)
    for feature_name in vertex_features:
        frame[feature_name] = dataclasses.I3VectorFloat(features[feature_name])
    
    ### Create offset lookups for the flattened feature arrays per event
    frame['NumberVertices'] = icetray.I3Int(len(features[features.keys()[0]]))
    #frame['Offset'] = icetray.I3Int(event_offset)
    event_offset += len(features[features.keys()[0]])

    ### Create coordinates for each vertex
    coordinate_matrix = np.vstack(coordinates.values()).T
    #print(coordinate_matrix.shape, len(frame[feature_name]))
    C_mean_centered, C_mean, C_std = normalize_coordinates(coordinate_matrix, weights=None, copy=True)
    C_cog_centered, C_cog, C_cog_std = normalize_coordinates(coordinate_matrix, weights=features['TotalCharge'], copy=True)
    frame['VertexX'] = dataclasses.I3VectorFloat(C_mean_centered[:, 0])
    frame['VertexY'] = dataclasses.I3VectorFloat(C_mean_centered[:, 1])
    frame['VertexZ'] = dataclasses.I3VectorFloat(C_mean_centered[:, 2])
    frame['COGShiftedVertexX'] = dataclasses.I3VectorFloat(C_cog_centered[:, 0])
    frame['COGShiftedVertexY'] = dataclasses.I3VectorFloat(C_cog_centered[:, 1])
    frame['COGShiftedVertexZ'] = dataclasses.I3VectorFloat(C_cog_centered[:, 2])

    ### Precompute pairwise distances for each vertex and both set of 
    distances = pairwise_distances(C_mean_centered).reshape(-1)
    distances_cog = pairwise_distances(C_cog_centered).reshape(-1)
    frame['Distances'] = dataclasses.I3VectorFloat(distances)
    frame['COGDistances'] = dataclasses.I3VectorFloat(distances_cog)
    #frame['DistancesOffset'] = icetray.I3Int(distances_offset)
    distances_offset += distances.shape[0] ** 2

    ### Compute targets for possible auxilliary tasks, i.e. position and direction of the interaction
    #print((nu.pos.x - C_mean[0]) / C_std[0], C_mean[0], C_std[0])
    frame['PrimaryX'] = dataclasses.I3Double((nu.pos.x - C_mean[0]) / C_std[0])
    frame['PrimaryY'] = dataclasses.I3Double((nu.pos.y - C_mean[1]) / C_std[1])
    frame['PrimaryZ'] = dataclasses.I3Double((nu.pos.z - C_mean[2]) / C_std[2])
    frame['COGPrimaryX'] = dataclasses.I3Double((nu.pos.x - C_cog[0]) / C_cog_std[0])
    frame['COGPrimaryY'] = dataclasses.I3Double((nu.pos.y - C_cog[1]) / C_cog_std[1])
    frame['COGPrimaryZ'] = dataclasses.I3Double((nu.pos.z - C_cog[2]) / C_cog_std[2])
    frame['PrimaryAzimuth'] = dataclasses.I3Double(nu.dir.azimuth)
    frame['PrimaryZenith'] = dataclasses.I3Double(nu.dir.zenith)

    ### Compute possible reco inputs that apply to entire event sets
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    frame['RecoX'] = dataclasses.I3Double((track_reco.pos.x - C_mean[0]) / C_std[0])
    frame['RecoY'] = dataclasses.I3Double((track_reco.pos.y - C_mean[1]) / C_std[1])
    frame['RecoZ'] = dataclasses.I3Double((track_reco.pos.z - C_mean[2]) / C_std[2])
    frame['COGRecoX'] = dataclasses.I3Double((track_reco.pos.x - C_cog[0]) / C_cog_std[0])
    frame['COGRecoY'] = dataclasses.I3Double((track_reco.pos.x - C_cog[1]) / C_cog_std[1])
    frame['COGRecoZ'] = dataclasses.I3Double((track_reco.pos.x - C_cog[2]) / C_cog_std[2])
    frame['RecoAzimuth'] = dataclasses.I3Double(track_reco.dir.azimuth)
    frame['RecoZenith'] = dataclasses.I3Double(track_reco.dir.zenith)
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
    tray.AddModule(get_weight_by_flux, 'get_weight_by_flux')
    tray.AddModule(I3TableWriter, 'I3TableWriter', keys = vertex_features + [
        # Meta data
        'PDGEncoding', 'InteractionType', 'NumberChannels', 'NeutrinoEnergy', 
        'CascadeEnergy', 'MuonEnergy', 'TrackLength', 'DeltaLLH', 'DCFiducialPE',
        # Lookups
        'NumberVertices',
        # Coordinates and pairwise distances
        'VertexX', 'VertexY', 'VertexZ', 'COGShiftedVertexX', 
        'COGShiftedVertexY', 'COGShiftedVertexZ', 'Distances',
        'COGDistances',
        # Auxilliary targets
        'PrimaryX', 'PrimaryY', 'PrimaryZ', 'COGPrimaryX', 'COGPrimaryY', 
        'COGPrimaryZ', 'PrimaryAzimuth', 'PrimaryZenith',
        # Reconstruction
        'RecoX', 'RecoY', 'RecoZ', 'COGRecoX', 'COGRecoY', 'COGRecoZ',
        'RecoAzimuth', 'RecoZenith',
        # Flux weights
        'NuMuFlux', 'NueFlux', 'NoFlux',
        ], 
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
    outfile = '/project/6008051/fuchsgru/data/data_dragon6_parts/{0}.hd5'.format(file_idx)
    create_dataset(outfile, [paths[file_idx]])

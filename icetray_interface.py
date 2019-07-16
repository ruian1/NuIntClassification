import numpy as np
import json
import torch
from create_dataset import create_dataset
import util

# Define the model architecture and parameters
MODEL_ARCHITECTURE = './training/hd5-3842795023/config.json'
MODEL_PARAMETERS = './training/hd5-3842795023/model_17'

with open('default_settings.json') as f:
    config = json.load(f)
with open(MODEL_ARCHITECTURE) as f:
    util.dict_update(config, json.load(f))

# Load the model
model = util.model_from_config(config)
if torch.cuda.is_available():
        model = model.cuda()
model.load_state_dict(torch.load(MODEL_PARAMETERS))

# Setup feature columns for the feature matrix, coordinate matrix and graph feature matrix
feature_columns = config['dataset']['features']
graph_feature_columns = config['dataset']['graph_features']
coordinate_columns = config['dataset']['coordinates']


def classify_event(frame):
    """ Icetray interface method to be run on a frame. It predicts a trackness score for each event.
    
    Parameters:
    -----------
    frame : I3Frame
        The frame which contains an event to classify.

    Returns:
    --------
    trackness : float
        The classification score.
    """
    # Get the features for the vertices
    features, coordinates, _ = create_dataset.get_events_from_frame(frame, charge_scale=charge_scale, time_scale=time_scale)
    
    ### Create coordinates for each vertex
    C = np.vstack(coordinates.values()).T
    C, C_mean, C_std = create_dataset.normalize_coordinates(C, weights=None, copy=True)
    C_cog, C_mean_cog, C_std_cog = create_dataset.normalize_coordinates(C, weights=features['TotalCharge'], copy=True)
    features['VertexX'] = C[:, 0]
    features['VertexY'] = C[:, 1]
    features['VertexZ'] = C[:, 2]
    features['COGCenteredVertexX'] = C_cog[:, 0]
    features['COGCenteredVertexY'] = C_cog[:, 1]
    features['COGCenteredVertexZ'] = C_cog[:, 2]

    # Get the features for the graph
    track_reco = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_Track']
    graph_features = {
        'RecoX' : (track_reco.pos.x - C_mean[0]) / C_std[0],
        'RecoY' : (track_reco.pos.y - C_mean[1]) / C_std[1],
        'RecoZ' : (track_reco.pos.z - C_mean[2]) / C_std[2],
        'COGCenteredRecoX' : (track_reco.pos.x - C_mean_cog[0]) / C_std_cog[0],
        'COGCenteredRecoY' : (track_reco.pos.y - C_mean_cog[1]) / C_std_cog[1],
        'COGCenteredRecoZ' : (track_reco.pos.z - C_mean_cog[2]) / C_std_cog[2],
        'RecoAzimuth' : track_reco.dir.azimuth,
        'RecoZenith' : track_reco.dir.zenith,
    }

    feature_columns = config

    # Create the feature matrix
    X = np.zeros((1, C.shape[0], len(feature_columns)))
    for feature_idx, feature in enumerate(feature_columns):
        X[0, :, feature_idx] = features[feature]
    C = np.zeros((1, C.shape[0], len(coordinate_columns)))
    for coordinate_idx, coordinate in enumerate(coordinate_columns):
        C[0, :, coordinate_idx] = features[coordinate]
    F = np.zeros((1, len(graph_feature_columns))):
    for graph_feature_idx, graph_feature in enumerate(graph_feature_columns):
        F[0, graph_feature_idx] = graph_features[graph_feature]
    masks = np.ones((1, C.shape[0], C.shape[0]))
    
    # Make torch tensors
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    X = torch.FloatTensor(X).to(device)
    C = torch.FloatTensor(C).to(device)
    F = torch.FloatTensor(F).to(device)
    masks = torch.FloatTensor(masks).to(device)

    model.eval()
    score = model(X, C, masks, F).data.cpu().numpy().squeeze()[0]
    return score






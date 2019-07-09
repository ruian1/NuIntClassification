# Documentation of the code and conducted experiements

## Dataset

The dataset is generated in multiple steps. The original source is an i3 file that contains detector data. From this i3 file a vertex attributed graph is generated for each event. In these events a DOM represents a vertex, while the graph itself is densly connected, meaning that each DOM is connected to each other DOM via an edge, the strength of which corresponds to the spatial distance between the DOMs.

### Vertex Features

For each DOM the following attributes are extracted and used to train the network:
- Charge of the first, last and maximal pulse
- Time of the first, last and maximal (w.r.t. charge) pulse
- Standard Devation of Pulse Times
- Coordinates of the DOM (x, y, z)

Time and charge values are scaled to approximately fit a [0, 1] range.

For time values a second set is generated in the following way: Using the track reconstruction (i.e. assuming it was true), one can calculate the time when a DOM is expected to register Cherenkov light.



The coordinates are centered arround their mean (possibly charge weighted) and scaled by an empircal value of 50m to resemble an approximate standard devation of ~1. Note that the same transformation is applied to the reconstruction.

### Graph Features

For each event the reconstruction is extracted as a graph-wise feature. These include the position (x, y, z) and angles (zenith and azimuth) as well as the length 

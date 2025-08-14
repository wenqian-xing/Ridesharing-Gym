# NYC Rideshare Simulation Data

This directory contains the historical NYC taxi and geographic data required for the rideshare simulation.

## ⚠️ Large Data Files Notice

The main data files are not included in the git repository due to size constraints (>200MB total). 

**To get the data files:**
```bash
cd data
python download_data.py
```

## Data Files

### `manhattan-trips.parquet` (68MB)
- **Content**: 2.98 million historical NYC taxi ride records (2008-2024)
- **Source**: NYC Taxi & Limousine Commission
- **Columns**:
  - `t`: Unix timestamp of ride request
  - `tpep_pickup_datetime`: Human-readable pickup datetime
  - `tpep_dropoff_datetime`: Human-readable dropoff datetime
  - `pickup_idx`: Pickup location node index (0-4332)
  - `dropoff_idx`: Dropoff location node index (0-4332)
  - `pickup_osmid`: OpenStreetMap pickup node ID
  - `dropoff_osmid`: OpenStreetMap dropoff node ID
  - `trip_distance`: Trip distance in units
  - `fare_amount`: Fare amount in dollars
  - `passenger_count`: Number of passengers

### `manhattan-distances.npy` (143MB)
- **Content**: 4333×4333 distance matrix for Manhattan street network
- **Format**: NumPy array (float64)
- **Structure**: `distances[i][j]` = shortest path distance from node i to node j
- **Units**: Distance units (proportional to travel time)
- **Source**: OpenStreetMap Manhattan road network

### `manhattan-nodes.parquet` (162KB)
- **Content**: Geographic coordinates for Manhattan street network nodes
- **Records**: 4,426 nodes with coordinates and OpenStreetMap IDs
- **Columns**:
  - `idx`: Node index (0-4425) for internal referencing
  - `lng`: Longitude coordinate
  - `lat`: Latitude coordinate  
  - `osmid`: OpenStreetMap node ID
- **Coverage**: Complete Manhattan street network nodes

### `taxi-zones.parquet` (36KB)
- **Content**: Mapping from street network nodes to NYC taxi zones
- **Records**: 4,333 node-to-zone mappings
- **Columns**:
  - `osmid`: OpenStreetMap node ID (matches manhattan-nodes.parquet)
  - `zone`: NYC taxi zone name (e.g., "Upper East Side North", "Central Park")
- **Purpose**: Geographic analysis and location-based aggregation

## Usage

The simulation automatically loads these files using relative paths:

```python
# Files are loaded automatically by the simulation
from environments.rideshare import RideshareEnvironment

env = RideshareEnvironment(
    n_cars=300,
    n_events=500000  # Uses up to 500K events from manhattan-trips.parquet
)
```

## Data Processing

The simulation processes the data as follows:

1. **Load trip data**: `pd.read_parquet("data/manhattan-trips.parquet")`
2. **Sort by timestamp**: Events processed in chronological order
3. **Normalize timestamps**: Start from t=0 for simulation
4. **Create event sequence**: JAX arrays for efficient processing
5. **Load distance matrix**: `np.load("data/manhattan-distances.npy")`

## Geographic Coverage

- **Street Network**: 4,333 distance matrix nodes covering Manhattan
- **Coordinate Data**: 4,426 nodes with precise lat/lng coordinates  
- **Zone Mapping**: 4,333 nodes mapped to NYC taxi zones
- **Coverage**: Complete Manhattan road network from OpenStreetMap
- **Pickup locations**: 4,308 unique nodes used as pickup points
- **Dropoff locations**: 4,333 unique nodes used as dropoff points

## Temporal Coverage

- **Time span**: 16 years (2008-2024)
- **Event density**: ~3,781 events/hour in peak periods
- **Recent data**: Higher density in recent years (2024)
- **Realistic patterns**: Preserves natural demand fluctuations

## File Integrity

Original data sources with verification:
- Trip data: MD5 `653f0d7d28348a3e998fdb38ef00ef47`
- Distance matrix: MD5 `95fda63cbed95bdb094f3b76baa7c7b4`

## Memory Usage

When loaded into memory:
- Trip data: ~553MB (full dataset with all columns)
- Distance matrix: ~143MB (4333² × 8 bytes)
- Geographic nodes: ~1.2MB (coordinates and IDs)
- Zone mapping: ~0.6MB (node-to-zone mapping)
- Total RAM usage: ~698MB for complete dataset
import torch
from torch.utils.data import IterableDataset
import xarray as xr
import numpy as np
import gc
import zarr
# from helpers import debug_print, set_device

class ZarrDataset(IterableDataset):
    def __init__(self, config):
        """
        Args:
            config: Configuration (e.g., OmegaConf dict) with keys:
              - zarr_file: path to zarr store
              - batch_size: number of points per mini-batch
        """
        debug_print()
        self.zarr_file = config.zarr_store
        self.batch_size = config.batch_size
        self.device = set_device()
        self.num_bands = 16
        self.ds = xr.open_zarr(self.zarr_file).isel(t=0)
        self.num_lat = len(self.ds.lat)
        self.num_lon = len(self.ds.lon)
        self.spatial_dim = self.num_lat * self.num_lon
        self.band_names = ['CMI_C01', 'CMI_C02', 'CMI_C03', 'CMI_C04', 'CMI_C05', 'CMI_C06', 'CMI_C07', 'CMI_C08', 'CMI_C09', 'CMI_C10',
                 'CMI_C11', 'CMI_C12', 'CMI_C13', 'CMI_C14', 'CMI_C15', 'CMI_C16']

        # Total points per time step
        self.points = self.num_lat * self.num_lon * self.num_bands
        self.batches = int(np.ceil(self.points / self.batch_size))

        # Precompute static spatial grid and pressure array (on CPU)
        self._precompute_static_data()
        
        
    def _precompute_static_data(self):
        """Precompute the spatial grid from latitude and longitude and the pressure values."""
        lats = self.ds.lat.values.astype(np.float32)  # shape (num_lat,)
        lons = self.ds.lon.values.astype(np.float32)  # shape (num_lon,)
        bands = np.arange(0, self.num_bands) # shape (num_bands, )

        # Create 2D grid for spatial coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")  # shape (num_lon, num_lat)
        self.lat_lon_index = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))  # shape (num_lat * num_lon, 2)

        # Convert longitudes from [0, 360] to [-180, 180]
        lon_centered = lon_grid - 180.0

        # Convert to radians
        lat_rad = np.deg2rad(lat_grid)
        lon_rad = np.deg2rad(lon_centered)

        # Compute Cartesian coordinates (spherical to Cartesian on unit sphere)
        cos_lat = np.cos(lat_rad)
        x = (cos_lat * np.cos(lon_rad)).astype(np.float32)
        y = (cos_lat * np.sin(lon_rad)).astype(np.float32)
        z = np.sin(lat_rad).astype(np.float32)

        # Flatten the 2D spatial grid (order: C-order, so that the fastest axis is longitude)
        x_flat = x.ravel()  # shape: (num_lat * num_lon,)
        y_flat = y.ravel()
        z_flat = z.ravel()
        # Tile the spatial coordinates for each pressure level.
        # Each time step has total points = num_pressure * num_lat * num_lon.
        self.static_grid = {
            "x": torch.from_numpy(np.tile(x_flat, self.num_bands)), # shape (num_lat * num_lon * num_bands,)
            "y": torch.from_numpy(np.tile(y_flat, self.num_bands)), 
            "z": torch.from_numpy(np.tile(z_flat, self.num_bands))  
        }

        # Repeat band values
        self.static_bands = torch.from_numpy(np.repeat(bands, self.spatial_dim)) # shape (num_lat * num_lon * num_bands, )

    def _create_batch(self, batch_idx):
        """Construct a mini-batch from loaded time_data."""
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.points)

        # Inputs: [x, y, z, bands]
        # Static grid data are precomputed
        x_batch = self.static_grid["x"][start:end]
        y_batch = self.static_grid["y"][start:end]
        z_batch = self.static_grid["z"][start:end]
        bands_batch = self.static_bands[start:end]
        inputs = torch.vstack([x_batch, y_batch, z_batch, bands_batch])

        wrap_indices = np.arange(start, end) % self.spatial_dim
        lat, lon = zip(*np.take(self.lat_lon_index, wrap_indices, axis=0))
        target = self._get_batch_target(start, end, list(lat), list(lon))

        return {"inputs": inputs, "target": target}
    
    def _get_batch_target(self, start, end, lats, lons):
        start_band_idx = start // self.spatial_dim
        end_band_idx = (end - 1) // self.spatial_dim
        target = []
        coord_start = 0
        coord_end = 0
        for i in range(start_band_idx, end_band_idx + 1):
            band = self.band_names[i]
            band_start = max(0, start - self.spatial_dim * i)
            band_end = min(self.spatial_dim, end - self.spatial_dim * i)
            coord_end = band_end - band_start
            for c in range(coord_start, coord_end):
                target.append(self.ds[band].sel(lat=lats[c], lon=lons[c]).values)
            coord_start = coord_end
            
        return np.array(target)

    def __iter__(self):
        """Iterate over each time step and yield mini-batches."""
        debug_print()
        for batch_idx in range(self.batches):
            yield self._create_batch(batch_idx)
            # Optional: clear GPU cache after each time step
            # torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def __len__(self):
        """Total number of mini-batches across all time steps."""
        return self.batches
"""
GOES Satellite Data to Zarr Conversion Pipeline

Modularized implementation for processing GOES-R series ABI L2 data into Zarr format.
"""

import zarr # 3.0.4
import xarray as xr # 2025.1.2
import numpy as np
import pyproj
import xesmf as xe
import warnings
import concurrent.futures
import pandas as pd
import argparse
from typing import List, Dict, Tuple
warnings.filterwarnings("ignore")

class SatelliteConfig:

    # Target Bands
    BANDS = ['CMI_C01', 'CMI_C02', 'CMI_C03', 'CMI_C04', 'CMI_C05', 'CMI_C06', 'CMI_C07', 'CMI_C08', 'CMI_C09', 'CMI_C10',
             'CMI_C11', 'CMI_C12', 'CMI_C13', 'CMI_C14', 'CMI_C15', 'CMI_C16']

    # Band specific metadata
    UNIVERSAL_BAND_METADATA = {
        'CMI_C01': {'central_band_wavelength': '0.47', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'blue', 'valid_range': [0.0, 1.0], 'units': '1', "grid_mapping": "crs"},

        'CMI_C02': {'central_band_wavelength': '0.64', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'red', 'valid_range': [0.0, 1.0], 'units': '1', "grid_mapping": "crs"},

        'CMI_C03': {'central_band_wavelength': '0.86', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'vegetation', 'valid_range': [0.0, 1.0], 'units': '1', "grid_mapping": "crs"},

        'CMI_C04': {'central_band_wavelength': '1.37', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'cirrus', 'valid_range': [0.0, 1.0], 'units': '1', "grid_mapping": "crs"},

        'CMI_C05': {'central_band_wavelength': '1.61', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'snow/ice', 'valid_range': [0.0, 1.0], 'units': '1', "grid_mapping": "crs"},

        'CMI_C06': {'central_band_wavelength': '2.24', 'long_name': 'ABI Cloud and Moisture Imagery reflectance factor',
                    'standard_name': 'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                    'descriptive_name': 'cloud particle size', 'valid_range': [0.0, 1.0], 'units': '1',
                    "grid_mapping": "crs"},

        'CMI_C07': {'central_band_wavelength': '3.90',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'shortwave window',
                    'valid_range': [197.31, 411.86], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C08': {'central_band_wavelength': '6.19',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'upper-level water vapour',
                    'valid_range': [138.05, 311.06], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C09': {'central_band_wavelength': '6.93',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'mid-level water vapour',
                    'valid_range': [137.7, 311.08], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C10': {'central_band_wavelength': '7.34',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'low-level water vapour',
                    'valid_range': [126.91, 331.2], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C11': {'central_band_wavelength': '8.44',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'cloud-top phase',
                    'valid_range': [127.69, 341.3], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C12': {'central_band_wavelength': '9.61',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'ozone',
                    'valid_range': [117.49, 311.06], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C13': {'central_band_wavelength': '10.33',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'clean longwave window',
                    'valid_range': [89.62, 341.27], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C14': {'central_band_wavelength': '11.21',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'longwave window',
                    'valid_range': [96.19, 341.28], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C15': {'central_band_wavelength': '12.29',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'dirty longwave window',
                    'valid_range': [97.38, 341.28], 'units': 'K', "grid_mapping": "crs"},

        'CMI_C16': {'central_band_wavelength': '13.28',
                    'long_name': 'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                    'standard_name': 'toa_brightness_temperature', 'descriptive_name': 'carbon dioxide',
                    'valid_range': [92.7, 318.26], 'units': 'K', "grid_mapping": "crs"}
    }

    # Satellite specific metadata - GOES 16
    GOES16_METADATA = {
        # Original attributes (retained)
        'naming_authority': 'gov.nesdis.noaa',
        'Conventions': 'CF-1.7',
        'institution': 'DOC/NOAA/NESDIS > U.S. Department of Commerce, National Oceanic and Atmospheric Administration, National Environmental Satellite, Data, and Information Services',
        'project': 'GOES',
        'platform_ID': 'G16',
        'instrument_type': 'GOES-R Series Advanced Baseline Imager (ABI)',
        'spatial_resolution': '0.02 degrees',
        'title': 'ABI L2 Cloud and Moisture Imagery (Regridded to WGS84)',
        'summary': 'Bilinear regridded multiple reflectance and emissive channel Cloud and Moisture Imagery Products are digital maps of clouds, moisture, and atmospheric windows at visible, near-IR, and IR bands.',
        'license': 'Public Domain',
        'processing_level': 'L2 regridded using xemsf [https://xesmf.readthedocs.io/en/stable/] to lat/lon grid'
    }

    # Satellite specific metadata - GOES 18
    GOES18_METADATA = {
        # Original attributes (retained)
        'naming_authority': 'gov.nesdis.noaa',
        'Conventions': 'CF-1.7',
        'institution': 'DOC/NOAA/NESDIS > U.S. Department of Commerce, National Oceanic and Atmospheric Administration, National Environmental Satellite, Data, and Information Services',
        'project': 'GOES',
        'platform_ID': 'G18',
        'instrument_type': 'GOES-R Series Advanced Baseline Imager (ABI)',
        'spatial_resolution': '0.02 degrees',
        'title': 'ABI L2 Cloud and Moisture Imagery (Regridded to WGS84)',
        'summary': 'Bilinear regridded multiple reflectance and emissive channel Cloud and Moisture Imagery Products are digital maps of clouds, moisture, and atmospheric windows at visible, near-IR, and IR bands.',
        'license': 'Public Domain',
        'processing_level': 'L2 regridded using xemsf [https://xesmf.readthedocs.io/en/stable/] to lat/lon grid'
    }

    # Cartopy mapping parameters
    CRS_METADATA = {
        "grid_mapping_name": "latitude_longitude",
        "longitude_of_prime_meridian": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "horizontal_datum_name": "WGS84",
        "reference_ellipsoid_name": "WGS84",
        "epsg_code": "EPSG:4326"
    }

    # Discarded variables
    UNUSED_DS_VARIABLES = ['goes_imager_projection', 'nominal_satellite_subpoint_lat', 'nominal_satellite_subpoint_lon', 'nominal_satellite_height', 'geospatial_lat_lon_extent', 'band_wavelength_C01', 'band_wavelength_C02',
                           'band_wavelength_C03', 'band_wavelength_C04', 'band_wavelength_C05', 'band_wavelength_C06', 'band_wavelength_C07', 'band_wavelength_C08', 'band_wavelength_C09', 'band_wavelength_C10', 'band_wavelength_C11',
                           'band_wavelength_C12', 'band_wavelength_C13', 'band_wavelength_C14', 'band_wavelength_C15', 'band_wavelength_C16', 'band_id_C01', 'band_id_C02', 'band_id_C03', 'band_id_C04', 'band_id_C05', 'band_id_C06',
                           'band_id_C07', 'band_id_C08', 'band_id_C09', 'band_id_C10', 'band_id_C11', 'band_id_C12', 'band_id_C13', 'band_id_C14', 'band_id_C15', 'band_id_C16', 'outlier_pixel_count_C01', 'min_reflectance_factor_C01',
                           'max_reflectance_factor_C01', 'mean_reflectance_factor_C01', 'std_dev_reflectance_factor_C01', 'outlier_pixel_count_C02', 'min_reflectance_factor_C02', 'max_reflectance_factor_C02', 'mean_reflectance_factor_C02',
                           'std_dev_reflectance_factor_C02', 'outlier_pixel_count_C03', 'min_reflectance_factor_C03', 'max_reflectance_factor_C03', 'mean_reflectance_factor_C03', 'std_dev_reflectance_factor_C03', 'outlier_pixel_count_C04',
                           'min_reflectance_factor_C04', 'max_reflectance_factor_C04', 'mean_reflectance_factor_C04', 'std_dev_reflectance_factor_C04', 'outlier_pixel_count_C05', 'min_reflectance_factor_C05', 'max_reflectance_factor_C05',
                           'mean_reflectance_factor_C05', 'std_dev_reflectance_factor_C05', 'outlier_pixel_count_C06', 'min_reflectance_factor_C06', 'max_reflectance_factor_C06', 'mean_reflectance_factor_C06', 'std_dev_reflectance_factor_C06',
                           'outlier_pixel_count_C07', 'min_brightness_temperature_C07', 'max_brightness_temperature_C07', 'mean_brightness_temperature_C07', 'std_dev_brightness_temperature_C07', 'outlier_pixel_count_C08',
                           'min_brightness_temperature_C08', 'max_brightness_temperature_C08', 'mean_brightness_temperature_C08', 'std_dev_brightness_temperature_C08', 'outlier_pixel_count_C09', 'min_brightness_temperature_C09',
                           'max_brightness_temperature_C09', 'mean_brightness_temperature_C09', 'std_dev_brightness_temperature_C09', 'outlier_pixel_count_C10', 'min_brightness_temperature_C10', 'max_brightness_temperature_C10',
                           'mean_brightness_temperature_C10', 'std_dev_brightness_temperature_C10', 'outlier_pixel_count_C11', 'min_brightness_temperature_C11', 'max_brightness_temperature_C11', 'mean_brightness_temperature_C11',
                           'std_dev_brightness_temperature_C11', 'outlier_pixel_count_C12', 'min_brightness_temperature_C12', 'max_brightness_temperature_C12', 'mean_brightness_temperature_C12', 'std_dev_brightness_temperature_C12',
                           'outlier_pixel_count_C13', 'min_brightness_temperature_C13', 'max_brightness_temperature_C13', 'mean_brightness_temperature_C13', 'std_dev_brightness_temperature_C13', 'outlier_pixel_count_C14',
                           'min_brightness_temperature_C14', 'max_brightness_temperature_C14', 'mean_brightness_temperature_C14', 'std_dev_brightness_temperature_C14', 'outlier_pixel_count_C15', 'min_brightness_temperature_C15',
                           'max_brightness_temperature_C15', 'mean_brightness_temperature_C15', 'std_dev_brightness_temperature_C15', 'outlier_pixel_count_C16', 'min_brightness_temperature_C16', 'max_brightness_temperature_C16',
                           'mean_brightness_temperature_C16', 'std_dev_brightness_temperature_C16', 'percent_uncorrectable_GRB_errors', 'percent_uncorrectable_L0_errors', 'dynamic_algorithm_input_data_container',
                           'algorithm_product_version_container', 'DQF_C01', 'DQF_C02', 'DQF_C03', 'DQF_C04', 'DQF_C05', 'DQF_C06', 'DQF_C07', 'DQF_C08', 'DQF_C09', 'DQF_C10', 'DQF_C11', 'DQF_C12','DQF_C13', 'DQF_C14', 'DQF_C15',
                           'DQF_C16']

#
# Core Processing Class
#

class GOESProcessor:
    def __init__(self, satellite: str, config: SatelliteConfig = SatelliteConfig(), compressors="auto"):
        self.config = config
        self.regridder = None
        self.zarr_store = None
        self.root_group = None
        self.compressors = compressors
        self.proj_params = {
            'a': 6378137.0,
            'b': 6356752.31414,
            'h': 35786023.0,
            'sweep': "x",
            'lon_0': -137.0 if satellite.lower() in ('west', 'goes18', '18') else -75.0
        }


    # --------------------------
    # Coordinate Handling
    # --------------------------

    def create_geostationary_projection(self, satellite: str) -> pyproj.Proj:
        """Create geostationary projection based on satellite"""
        geos_proj = pyproj.Proj(proj='geos', **self.proj_params)
        return geos_proj

    def calculate_geocoordinates(self, dataset: xr.Dataset, satellite: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate latitude/longitude coordinates from geostationary projection"""
        proj = self.create_geostationary_projection(satellite)
        x_rad, y_rad = np.meshgrid(dataset.x.values, dataset.y.values)
        x_metres = x_rad * self.proj_params['h']
        y_metres = y_rad * self.proj_params['h']

        lons, lats = proj(x_metres, y_metres, inverse=True)
        return lons, lats

    def add_coordinates(self, dataset: xr.Dataset, lons: np.ndarray, lats: np.ndarray) -> xr.Dataset:
        """Add latitude and longitude coordinates to dataset"""
        dataset.coords['lon'] = (('y', 'x'), lons)
        dataset.coords['lat'] = (('y', 'x'), lats)

        return dataset


    # --------------------------
    # Zarr Storage Management
    # --------------------------

    def initialize_zarr_store(self, store_name: str, metadata: Dict):
        """Initialize Zarr storage with metadata"""
        self.zarr_store = zarr.storage.LocalStore(store_name) # base store
        self.root_group = zarr.open_group(store=self.zarr_store) # base group
        for k, v in metadata.items():
            self.root_group.attrs[k]= v


    # --------------------------
    # Data Regridder
    # --------------------------

    def create_regridder(self, source_ds: xr.Dataset, target_ds: xr.Dataset, weights: str) -> xe.Regridder:
        """Initialize regridder with optional weights"""
        return xe.Regridder(
            source_ds,
            target_ds,
            method='bilinear',
            periodic=False,
            weights=weights
        )



    # --------------------------
    # Main Processing Pipeline
    # --------------------------

    def process_files(self, file_paths: List[str], store_name: str, target_grid_file: str,
                      regridder_weights: str = None, satellite: str = 'west',
                      chunks: Tuple[int] = (336, 256, 256), shards: int = None,
                      dt_units: str = "seconds since 2000-01-01T12:00:00", dt_calendar: str = "proleptic_gregorian",
                      log_batch=False):
        """

        :param file_paths: List of input NetCDF files
        :param store_name: Output Zarr store name
        :param target_grid: Target grid definition file
        :param regridder_weights: Precomputed regridder weights
        :param satellite: Satellite identifier ('west' or 'east')
        :param chunks: Zarr chunk dimensions
        :param batch_size: Number of files to process per batch
        """
        batch_size = chunks[0]

        # Initialize components
        sample_ds = xr.open_dataset(file_paths[0], chunks='auto', drop_variables=self.config.UNUSED_DS_VARIABLES)

        # Calculate geocoordinates
        lons, lats = self.calculate_geocoordinates(sample_ds, satellite)

        # Set coordinates
        sample_ds = self.add_coordinates(sample_ds, lons, lats)

        # Target grid
        target_ds = xr.open_dataset(target_grid_file, chunks='auto')

        # Initialize regridder
        self.regridder = self.create_regridder(sample_ds, target_ds, regridder_weights)

        sample_ds = self.regridder(sample_ds['CMI_C07'])
        mask = sample_ds.where(sample_ds < 197.3).values

        # Initialize Zarr store
        self.initialize_zarr_store(store_name, self.config.GOES18_METADATA if 'west' in satellite else self.config.GOES16_METADATA)

        # Create coordinate arrays
        if 'lat' not in self.root_group:
            lat = zarr.create_array(self.zarr_store, name='lat', shape=sample_ds['lat'].values.shape, dtype=np.float32,
                                   attributes={}, dimension_names=['lat'])
            lat[:] = sample_ds['lat'].values

        if 'lon' not in self.root_group:
            lon = zarr.create_array(self.zarr_store, name='lon', shape=sample_ds['lon'].values.shape, dtype=np.float32,
                                   attributes={}, dimension_names=['lon'])
            lon[:] = sample_ds['lon'].values

        def process_band(da, band):
            ds_regrid = self.regridder(da).chunk(chunks)

            # Apply mask: set values to NaN where mask is 0
            ds_regrid = ds_regrid.where(mask != 0, np.nan)

            if band in self.root_group:
                za = zarr.open_array(self.zarr_store, path=band)
                za.append(ds_regrid.values)
            else:
                za = zarr.create_array(self.zarr_store, name=band, shape=ds_regrid.values.shape, dtype=np.float32,
                                       attributes=self.config.UNIVERSAL_BAND_METADATA[band],
                                       chunks=chunks, dimension_names=['t', 'lat', 'lon'],
                                       compressors=self.compressors, shards=shards)
                za[:] = ds_regrid.values

        # Process in batches
        # iterate through each batch
        for i in range(0, len(file_paths), batch_size or chunks[0]):
            # Open and concatenate files
            ds = self.add_coordinates(xr.open_mfdataset(
                file_paths[i: min(len(file_paths), i + batch_size)],
                combine="nested",
                concat_dim="t",
                chunks='auto',
                drop_variables=self.config.UNUSED_DS_VARIABLES
            ), lats, lons)

            # Encode time values
            encoded_time, _, _ = xr.coding.times.encode_cf_datetime(ds['t'].values, units=dt_units,
                                                                    calendar=dt_calendar)
            # Write time values to Zarr
            if 't' in self.root_group:
                ta = zarr.open_array(self.zarr_store, path="t")
                ta.append(encoded_time)
            else:
                ta = zarr.create_array(self.zarr_store, name="t", shape=encoded_time.shape, dtype=np.float32,
                                       attributes={"units": dt_units, "calendar": dt_calendar}, dimension_names=['t'])
                ta[:] = encoded_time
            
            # Process bands
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(process_band, [ds[band] for band in self.config.BANDS], self.config.BANDS)
            if log_batch:
                print(f"Completed conversion of batch {i + 1}.")
        # Consolidate metadata
        zarr.consolidate_metadata(self.zarr_store)

def read_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path, usecols=["timestamp", "file_url"])
    df = df.sort_values(by="timestamp", ascending=True)
    return df["file_url"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process satellite data with regridding.")
    parser.add_argument("--goes_file_csv", type=str, help="Path to the CSV file containing lists of goes data with columns date and file_url. Only include files you want to convert.")
    parser.add_argument("--store_path", type=str, default="./goes_conversion.zarr", help="Directory store the output files, should be a new directory unless adding more data to new directory")
    parser.add_argument("--target_grid_file", type=str, help="Path to the target grid file.")
    parser.add_argument("--regridder_weight_file", type=str, help="Path to the regridding weight file.")
    parser.add_argument("--chunk_size", type=int, default=336, help="Chunk size across timesteps. Redundant if appending new data.")
    parser.add_argument("--satellite", type=str, choices=["west", "east"], help="Satellite name of the dataset trying to convert. Should be one of west (goes18) or east (goes16).")
    parser.add_argument("--log-batch", action="store_true", default=False, help="Log after completing a batch.")
    args = parser.parse_args([])

    file_paths = read_csv_file(args.goes_file_csv)
    config = SatelliteConfig()
    processor = GOESProcessor(args.satellite, config)

    processor.process_files(
        file_paths=file_paths,
        store_name=args.store_path,
        target_grid_file=args.target_grid_file,
        regridder_weights=args.regridder_weight_file,
        satellite=args.satellite,
        chunks=(args.chunk_size, 256, 256),
        log_batch=True
    )
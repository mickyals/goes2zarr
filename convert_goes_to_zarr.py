"""
GOES Satellite Data to Zarr Conversion Pipeline

Modularized implementation for processing GOES-R series ABI L2 data into Zarr format.
"""

from functools import partial
import logging
import pathlib
import zarr
import xarray as xr
import numpy as np
from numcodecs.zarr3 import PCodec
import pyproj
import xesmf as xe
import concurrent.futures
import argparse

LOGGER = logging.getLogger("goes2zarr")

UINT16_MAX = 65535

class SatelliteConfig:

    def __init__(self, include_data_quality_vars = False, include_all_vars = False):
        self._include_dq = include_all_vars or include_data_quality_vars
        self._include_all = include_all_vars

    _CMI_VARS = ['CMI_C01', 'CMI_C02', 'CMI_C03', 'CMI_C04', 'CMI_C05', 'CMI_C06', 'CMI_C07', 'CMI_C08', 'CMI_C09', 'CMI_C10',
                 'CMI_C11', 'CMI_C12', 'CMI_C13', 'CMI_C14', 'CMI_C15', 'CMI_C16']

    _DQF_VARS =  ['DQF_C01', 'DQF_C02', 'DQF_C03', 'DQF_C04', 'DQF_C05', 'DQF_C06', 'DQF_C07', 'DQF_C08', 'DQF_C09', 'DQF_C10', 'DQF_C11', 'DQF_C12','DQF_C13', 'DQF_C14', 'DQF_C15', 'DQF_C16'] # These may be the only useful ones

    _EXTRA_VARS = [ 'goes_imager_projection', 'nominal_satellite_subpoint_lat', 'nominal_satellite_subpoint_lon', 'nominal_satellite_height', 'geospatial_lat_lon_extent', 'band_wavelength_C01', 'band_wavelength_C02',
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
                    'algorithm_product_version_container']

    @property
    def BANDS(self):
        return self._CMI_VARS + \
            (self._DQF_VARS if self._include_dq else []) + \
            (self._EXTRA_VARS if self._include_all else [])


    @property
    def UNUSED_DS_VARIABLES(self):
        if self._include_all:
            return []
        if self._include_dq:
            return self._EXTRA_VARS
        return self._EXTRA_VARS + self._DQF_VARS


    @property
    def UNIVERSAL_BAND_METADATA(self):
        return self._CMI_UNIVERSAL_BAND_METADATA | \
                (self._DQF_UNIVERSAL_BAND_METADATA if self._include_dq else {}) | \
                (self._EXTRA_UNIVERSAL_BAND_METADATA if self._include_all else {})


    # Band specific metadata
    _CMI_UNIVERSAL_BAND_METADATA = {
        'CMI_C01':{'central_band_wavelength':'0.47', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle', 
                  'descriptive_name':'blue', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0},
        'CMI_C02':{'central_band_wavelength':'0.64', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                  'descriptive_name':'red', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0}, 
        'CMI_C03':{'central_band_wavelength':'0.86', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                  'descriptive_name':'vegetation', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0}, 
        'CMI_C04':{'central_band_wavelength':'1.37', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                  'descriptive_name':'cirrus', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0},
        'CMI_C05':{'central_band_wavelength':'1.61', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                  'descriptive_name':'snow/ice', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0},
        'CMI_C06':{'central_band_wavelength':'2.24', 'long_name':'ABI Cloud and Moisture Imagery reflectance factor',
                  'standard_name':'toa_lambertian_equivalent_albedo_multiplied_by_cosine_solar_zenith_angle',
                  'descriptive_name':'cloud particle size', 'valid_range':[0.0,1.0], 'units':'1', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 1/UINT16_MAX, "add_offset":0.0},
        'CMI_C07':{'central_band_wavelength':'3.90', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'shortwave window',
                  'valid_range':[197.31, 411.86], 'units':'K',"grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 214.55/UINT16_MAX, "add_offset":197.31},
        'CMI_C08':{'central_band_wavelength':'6.19', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'upper-level water vapour',
                  'valid_range':[138.05, 311.06], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 173.01/UINT16_MAX, "add_offset":138.05},
        'CMI_C09':{'central_band_wavelength':'6.93', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'mid-level water vapour',
                  'valid_range':[137.7 , 311.08], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 173.38/UINT16_MAX, "add_offset":137.7},
        'CMI_C10':{'central_band_wavelength':'7.34', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'low-level water vapour', 
                  'valid_range':[126.91, 331.2 ], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 204.29/UINT16_MAX, "add_offset":126.91},
        'CMI_C11':{'central_band_wavelength':'8.44', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'cloud-top phase',
                  'valid_range':[127.69, 341.3 ], 'units':'K',"grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 213.61/UINT16_MAX, "add_offset":127.69},
        'CMI_C12':{'central_band_wavelength':'9.61', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'ozone', 
                  'valid_range':[117.49, 311.06], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 193.57/UINT16_MAX, "add_offset":117.49},
        'CMI_C13':{'central_band_wavelength':'10.33', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'clean longwave window',
                  'valid_range':[ 89.62, 341.27], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 251.65/UINT16_MAX, "add_offset":89.62},
        'CMI_C14':{'central_band_wavelength':'11.21', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere', 
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'longwave window',
                  'valid_range':[ 96.19, 341.28], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 245.09/UINT16_MAX, "add_offset":96.19},
        'CMI_C15':{'central_band_wavelength':'12.29', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'dirty longwave window',
                  'valid_range':[ 97.38, 341.28], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 243.9/UINT16_MAX, "add_offset":97.38},
        'CMI_C16':{'central_band_wavelength':'13.28', 'long_name':'ABI Cloud and Moisture Imagery brightness temperature at top of atmosphere',
                  'standard_name':'toa_brightness_temperature', 'descriptive_name':'carbon dioxide',
                  'valid_range':[ 92.7 , 318.26], 'units':'K', "grid_mapping": "crs", "coordinates":"crs",
                  "scale_factor": 225.56/UINT16_MAX, "add_offset":92.7}
    }

    _DQF_UNIVERSAL_BAND_METADATA = {k: {"flag_values": [0, 1, 2, 3, 4], "flag_meanings": "good_pixel_qf conditionally_usable_pixel_qf out_of_range_pixel_qf no_value_pixel_qf focal_plane_temperature_threshold_exceeded_qf"} for k in _DQF_VARS}
    _EXTRA_UNIVERSAL_BAND_METADATA = {k: {} for k in _EXTRA_VARS}

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

    GOES16_LAT_LON_METADATA = {
        "lat": {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
            "valid_min": -81.3282,
            "valid_max": 81.3282,
            "comment": "WGS84 latitude grid for GOES-16 regridded data",
        },
        "lon" : {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
            "valid_min": -156.2995,  # Specific to GOES-16's domain
            "valid_max": 6.2995,
            "comment": "WGS84 longitude grid for GOES-16 (CONUS/Atlantic focus)",
        },
    }

    GOES18_LAT_LON_METADATA = {
        "lat": {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
            "valid_min": -81.3282,
            "valid_max": 81.3282,
            "comment": "WGS84 latitude grid for GOES-18 regridded data"
        },
        "lon": {
            "standard_name": "longitude",
            "long_name": "Longitude",
            "units": "degrees_east",
            "axis": "X",
            "valid_min": 141.7005,  # Adjusted for longitude wrapping
            "valid_max": 304.2995,
            "comment": "WGS84 longitude grid (values >180 wrapped to -180) for GOES-18"
        }
    }


#
# Core Processing Class
#

class GOESProcessor:
    def __init__(self, satellite: str, config: SatelliteConfig = SatelliteConfig(), compressors="auto", serializer="auto"):
        self.config = config
        self.regridder = None
        self.zarr_store = None
        self.root_group = None
        self.compressors = compressors
        self.serializer = serializer
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

    def calculate_geocoordinates(self, dataset: xr.Dataset, satellite: str) -> tuple[np.ndarray, np.ndarray]:
        """Calculate latitude/longitude coordinates from geostationary projection"""
        LOGGER.debug("Calculating geocoordinates")
        proj = self.create_geostationary_projection(satellite)
        x_rad, y_rad = np.meshgrid(dataset.x.values, dataset.y.values)
        x_metres = x_rad * self.proj_params['h']
        y_metres = y_rad * self.proj_params['h']

        lons, lats = proj(x_metres, y_metres, inverse=True)
        return lons, lats

    def add_coordinates(self, dataset: xr.Dataset, lons: np.ndarray, lats: np.ndarray) -> xr.Dataset:
        """Add latitude and longitude coordinates to dataset"""
        LOGGER.debug("Adding geocoordinates to dataset '%s'", dataset)
        dataset.coords['lon'] = (('y', 'x'), lons)
        dataset.coords['lat'] = (('y', 'x'), lats)

        return dataset


    # --------------------------
    # Zarr Storage Management
    # --------------------------

    def initialize_zarr_store(self, store_name: str, metadata: dict):
        """Initialize Zarr storage with metadata"""
        LOGGER.debug("Initialize zarr storage at '%s'", store_name)
        self.zarr_store = zarr.storage.LocalStore(store_name) # base store
        self.root_group = zarr.open_group(store=self.zarr_store) # base group
        for k, v in metadata.items():
            self.root_group.attrs[k]= v

    def process_band(self, band, ds, chunks, mask, shards):
        LOGGER.debug("Processing band '%s'", band)
        da = ds[band]
        ds_regrid = self.regridder(da).chunk(chunks)

        # Apply mask: set values to NaN where mask is 0
        ds_regrid = ds_regrid.where(mask, np.nan)
        
        # TODO: floor every DQF_CXX band (floor all values in band if the name of the band is DQF...)
        scale_factor = self.config.UNIVERSAL_BAND_METADATA[band]['scale_factor']
        add_offset = self.config.UNIVERSAL_BAND_METADATA[band]['add_offset']
        if band in self.root_group:
            za = zarr.open_array(self.zarr_store, path=band)
            za.append(np.rint((ds_regrid.values - add_offset) / scale_factor))
        else:
            za = zarr.create_array(self.zarr_store, name=band, shape=ds_regrid.values.shape, dtype=np.uint16,
                                    attributes=self.config.UNIVERSAL_BAND_METADATA[band],
                                    chunks=chunks, dimension_names=['t', 'lat', 'lon'],
                                    compressors=self.compressors, serializer=self.serializer, shards=shards)
            za[:] = np.rint((ds_regrid.values - add_offset) / scale_factor)

    # --------------------------
    # Main Processing Pipeline
    # --------------------------

    def process_files(self, file_paths: list[str], store_name: str, target_grid_file: str,
                      regridder_weights: str = None, satellite: str = 'west',
                      chunks: tuple[int] = (336, 256, 256), shards: int = None,
                      dt_units: str = "seconds since 2000-01-01T12:00:00", dt_calendar: str = "proleptic_gregorian"):
        """

        :param file_paths: List of input NetCDF files
        :param store_name: Output Zarr store name
        :param target_grid: Target grid definition file
        :param regridder_weights: Precomputed regridder weights
        :param satellite: Satellite identifier ('west' or 'east')
        :param chunks: Zarr chunk dimensions
        :param batch_size: Number of files to process per batch
        """
        LOGGER.debug("Starting processing netCDF input files")
        LOGGER.debug("Creating sample dataset to calculate initial extents and weights using netCDF file at '%s'", file_paths[0])
        # Initialize components
        sample_ds = xr.open_dataset(file_paths[0], chunks='auto', drop_variables=self.config.UNUSED_DS_VARIABLES)

        # Calculate geocoordinates
        lons, lats = self.calculate_geocoordinates(sample_ds, satellite)

        # Set coordinates
        sample_ds = self.add_coordinates(sample_ds, lons, lats)

        # Target grid
        target_ds = xr.open_dataset(target_grid_file, chunks='auto')

        # Initialize regridder
        self.regridder = xe.Regridder(
            sample_ds,
            target_ds,
            method='bilinear',
            periodic=False,
            weights=regridder_weights
        )

        sample_ds = self.regridder(sample_ds['CMI_C07'])
        mask = sample_ds >= 197.3

        # Initialize Zarr store
        self.initialize_zarr_store(store_name, self.config.GOES18_METADATA if 'west' in satellite else self.config.GOES16_METADATA)

        # Create coordinate arrays
        LOGGER.debug("Create coordinate arrays from sample dataset")
        if 'lat' not in self.root_group:
            lat = zarr.create_array(self.zarr_store, name='lat', shape=sample_ds['lat'].values.shape, dtype=np.float32,
                                    attributes=self.config.GOES18_LAT_LON_METADATA['lat'] if 'west' in satellite else self.config.GOES16_LAT_LON_METADATA['lat'],
                                    dimension_names=['lat'])
            lat[:] = sample_ds['lat'].values

        if 'lon' not in self.root_group:
            lon = zarr.create_array(self.zarr_store, name='lon', shape=sample_ds['lon'].values.shape, dtype=np.float32,
                                   attributes=self.config.GOES18_LAT_LON_METADATA['lon'] if 'west' in satellite else self.config.GOES16_LAT_LON_METADATA['lon'],
                                   dimension_names=['lon'])
            lon[:] = sample_ds['lon'].values
        
        batch_size = chunks[0]
        # Process in batches
        # iterate through each batch
        LOGGER.debug("Starting batch processing of netCDF files. Total files: %d Batch size: %d", len(file_paths), batch_size)
        for i in range(0, len(file_paths), batch_size):
            # Open and concatenate files
            batch_end = min(len(file_paths), i + batch_size)
            LOGGER.debug("Starting batch of files from '%s' to '%s' (indexes %d to %d)", file_paths[i], file_paths[batch_end - 1], i, batch_end)
            ds = self.add_coordinates(xr.open_mfdataset(
                file_paths[i: batch_end],
                combine="nested",
                concat_dim="t",
                chunks='auto',
                drop_variables=self.config.UNUSED_DS_VARIABLES
            ), lats, lons)

            # Encode time values
            encoded_time, _, _ = xr.coding.times.encode_cf_datetime(ds['t'].values, units=dt_units,
                                                                    calendar=dt_calendar, dtype=np.float64)
            # Write time values to Zarr
            if 't' in self.root_group:
                ta = zarr.open_array(self.zarr_store, path="t")
                ta.append(encoded_time)
            else:
                ta = zarr.create_array(self.zarr_store, name="t", shape=encoded_time.shape, dtype=np.float64,
                                       attributes={"units": dt_units, "calendar": dt_calendar, "standard_name": "time"},
                                       dimension_names=['t'])
                ta[:] = encoded_time

            # Process bands
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(partial(self.process_band, ds=ds, chunks=chunks, mask=mask, shards=shards),
                             self.config.BANDS)
            LOGGER.info("Completed conversion of batch from file %d to %d.", i, batch_end)
        # Consolidate metadata
        zarr.consolidate_metadata(self.zarr_store)

def read_input_files(file_path):
    LOGGER.debug("Reading input netCDF file list from: '%s'", file_path)
    with open(file_path) as f:
        return f.read().splitlines()

def get_target_grid_file(satellite, grid_lon_extent):
    LOGGER.debug("Finding target grid file for %s satellite and %s extent", satellite, grid_lon_extent)
    grid_dir = pathlib.Path(__file__).parent / "grids"
    if grid_lon_extent == "0-360":
        return grid_dir / f"ds_target_{satellite}_0_360.nc"
    grid_file = grid_dir / f"ds_target_{satellite}_-180_180.nc"
    LOGGER.info("Using grid file '%s'", grid_file)
    return grid_file


def configure_logger(verbosity_level):
    if verbosity_level < 1:
        log_level = logging.CRITICAL
    elif verbosity_level == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)


def check_weight_file(regridder_weight_file, satellite):
    if regridder_weight_file and (weight_file_path := pathlib.Path(regridder_weight_file)).is_file():
        if satellite not in weight_file_path.name.lower():
            LOGGER.warning("Regridder weight file '%s' may be for the wrong satellite (east vs. west). "
                            "Double check that you have selected the correct file.", regridder_weight_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process satellite data with regridding.")
    parser.add_argument("--goes-file-list", required=True, help="Path to a file containing a list of paths or URLs where the input netCDF GOES data can be found (one file per line).")
    parser.add_argument("--store-path", default="./goes_conversion.zarr", help="Directory store the output files, should be a new directory unless adding more data to an existing directory")
    parser.add_argument("--grid-lon-extent", default="0-360", choices=["0-360", "+-180"], help="Longitude range of the grid that data will be mapped to. Choices are 0 to 360 degrees or -180 to 180 degrees.")
    parser.add_argument("--regridder-weight-file", help="Path to the regridding weight file.")
    parser.add_argument("--temporal-chunk-size", default=24, type=int, help="Chunk size across timesteps. Redundant if appending new data.")
    parser.add_argument("--spatial-chunk-size", default=512, type=int, help="Chunk size across spatial extent. Redundant if appending new data.")
    parser.add_argument("--satellite", required=True, choices=["west", "east"], help="Satellite name of the dataset trying to convert. Should be one of west (goes18) or east (goes16).")
    parser.add_argument("--include-data-quality-vars", action="store_true", help="Include DQI_* variables in the output zarr files. Default is to only include CMI_* variables.")
    parser.add_argument("--include-all-vars", action="store_true", help="Include all variables (including DQI_* and others) in the output zarr files. Default is to only include CMI_* variables.")
    parser.add_argument("--compressor-level", type=int, default=9, help="Compression level for compression codec (Zstd) used to compress the zarr output data.")
    parser.add_argument("--serializer-level", type=int, default=9, help="Compression level for serializer codec (PCodec) used to compress the zarr output data.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Log verbose output")
    args = parser.parse_args()

    configure_logger(args.verbose)
    check_weight_file(args.regridder_weight_file, args.satellite)

    file_paths = read_input_files(args.goes_file_list)
    config = SatelliteConfig(args.include_data_quality_vars, args.include_all_vars)
    processor = GOESProcessor(args.satellite, config, compressors=zarr.codecs.ZstdCodec(level=args.compressor_level), serializer=PCodec(level=args.serializer_level))

    processor.process_files(
        file_paths=file_paths,
        store_name=args.store_path,
        target_grid_file=get_target_grid_file(args.satellite, args.grid_lon_extent),
        regridder_weights=args.regridder_weight_file,
        satellite=args.satellite,
        chunks=(args.temporal_chunk_size, args.spatial_chunk_size, args.spatial_chunk_size)
    )

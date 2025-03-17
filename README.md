# GOES2ZARR

Convert GOES netCDF files to zarr format.

## Requirements

### Local usage

- Python (3.12+)
- [ESMF](https://earthsystemmodeling.org/)

### Docker

- docker

## Installation

Download the precalculated regridding weights from [google drive](https://drive.google.com/drive/folders/1QpsCvb7x0cG9H4-qkeGjipVuKnUFt2Yj?usp=sharing) (optional). These weights are too large to be distributed
in this repository so they are made available from drive. They are not required to run goes2zarr but without
them, the script will recalculate the weights every time which will take a lot longer to run.

### Local usage

ESMF and python should already be installed on your machine.
Install python dependencies:

```sh
pip install -r requirements.txt
```

### Docker

Build the docker image locally:

```sh
docker build -t goes2zarr:latest .
```

## Run 

## Local usage

To see all command line options run:

```sh
python convert_goes_to_zarr.py --help
```

Example run (basic):

```sh
python convert_goes_to_zarr.py --goes-file-list /path/to/filelist/file --satellite west --regridder-weight-file /path/to/downloaded/goeswest_regridder.nc
```

## Docker

To see all command line options run:

```sh
docker run --rm goes2zarr:latest --help 
```

Example run (basic):

```sh
docker run -v /path/to/filelist/file:/goes-file-list:ro -v /path/to/downloaded/goeswest_regridder.nc:/goeswest_regridder.nc:ro -v ./tmp/:/result/:rw --rm goes2zarr:latest --goes-file-list /goes-file-list --satellite west --regridder-weight-file /goeswest_regridder.nc --store-path /result/result.zarr
```

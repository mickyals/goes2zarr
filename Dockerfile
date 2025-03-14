FROM continuumio/miniconda3:25.1.1-2 AS base

RUN conda install -y conda-forge::esmpy pip

COPY ./requirements.txt /requirements.txt

RUN /opt/conda/bin/pip install -r /requirements.txt

WORKDIR /app

COPY ./convert_goes_to_zarr.py /app/convert_goes_to_zarr.py

COPY ./grids/ /app/grids

ENV PATH=/opt/conda/bin:$PATH

ENTRYPOINT [ "python", "convert_goes_to_zarr.py" ]

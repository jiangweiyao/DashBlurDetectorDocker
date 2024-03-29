FROM continuumio/miniconda3

# Install the conda environment
COPY dash_environment.yml /
COPY blur_dash.py /
COPY model_conv_6classes.pth /
COPY class.txt /
COPY fullname.txt /
RUN conda env update --name base --file dash_environment.yml && conda clean -a

# Add conda installation dir to PATH (instead of doing 'conda activate')
ENV PATH /opt/conda/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name base > base.yml

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "blur_dash:server"]

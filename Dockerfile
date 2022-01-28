FROM jupyter/base-notebook

# expose jupyter notebook ports
EXPOSE 8082
EXPOSE 8083
EXPOSE 8888

USER root
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    git \
    htop \
    neovim

USER jovyan
COPY . /home/jovyan/modes
COPY docs/notebooks /home/jovyan/notebooks
RUN conda init bash

RUN pip install modes
WORKDIR /home/jovyan

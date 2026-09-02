# Dockerfile for dev container
FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin

RUN apt-get update

RUN apt-get install git-lfs

RUN git lfs install

Run pip3 install transformers==4.45.2

RUN pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt-get install graphviz graphviz-dev -y

# install R and some depedencies
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && apt-get install -y software-properties-common && add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' && apt update && apt install -y r-base && apt install -y build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev libtiff-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev pkg-config libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libwebp-dev libgit2-dev libsodium-dev libmagick++-dev libsecret-1-dev libsodium-dev libuv1-dev

# preparations for the SID package for R
RUN mkdir /home/apps && mkdir /home/apps/rlibs

COPY .docker/installer.R /home/apps/rlibs/installer.R

RUN Rscript /home/apps/rlibs/installer.R

RUN Rscript -e 'install.packages(c("usethis", "systemfonts", "textshaping", "pkgdown" ,"rcmdcheck", "roxygen2", "rversions", "urlchecker", "devtools"), dependencies = TRUE)'

# finally, install SID
RUN Rscript -e 'require(devtools); devtools::install_version("SID",version="1.0")'

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install bnlearn==0.4.0 scipy==1.5.0

RUN pip3 install --upgrade bnlearn scipy causaldag

RUN pip3 install GPy igraph gcastle

WORKDIR "/workspaces/tagging_causality"
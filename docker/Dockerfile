FROM nvcr.io/nvidia/pytorch:22.08-py3


RUN apt-get update && apt-get install -y \
    curl \
    python3-pip \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*




# Create a working directory
RUN mkdir /app
WORKDIR /app


ARG UNAME=jieqiu
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m ${USER} -u $UID -g $GID $UNAME
RUN echo "$UNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$UNAME
USER $UNAME




# Usage
RUN pip3 install tensorboard imageio albumentations
RUN pip3 install git+https://github.com/rwightman/pytorch-image-models

CMD /bin/bash
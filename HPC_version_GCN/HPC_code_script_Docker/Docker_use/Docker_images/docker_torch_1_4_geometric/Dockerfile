ARG  CODE_VERSION=1.4-cuda10.1-cudnn7-devel

FROM pytorch/pytorch:${CODE_VERSION}

LABEL MAINTAINER="Xiang Li <xiangli13257@gmail.com>"

ENV CPATH=/usr/local/cuda/include:$CPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY docker-entrypoint.sh /usr/local/bin/

RUN apt-get update && apt-get install -y libmetis-dev \
	&& pip install \
 	pandas \
	scipy \
	metis \
	seaborn \
	matplotlib \
	&& pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
	&& pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html  \
	&& pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
	&& pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
	&& pip install torch-geometric \
	&& pip uninstall -y networkx \
	&& pip install -I networkx==2.3.0

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

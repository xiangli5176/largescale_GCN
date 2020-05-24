ARG  CODE_VERSION=1.2-cuda10.0-cudnn7-devel
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
	&& pip install --verbose --no-cache-dir \
	torch-scatter \
	torch-sparse \
	torch-cluster \
	torch-spline-conv \
	&& pip install \
	torch-geometric \
	&& pip uninstall -y networkx \
	&& pip install -I networkx==2.3.0

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

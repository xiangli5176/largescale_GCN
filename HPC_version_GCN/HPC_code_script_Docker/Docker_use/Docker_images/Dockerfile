ARG  CODE_VERSION=1.1.0-cuda10.0-cudnn7.5-devel
FROM pytorch/pytorch:${CODE_VERSION}

LABEL MAINTAINER="Xiang Li <xiangli13257@gmail.com>"

ENV CPATH=/usr/local/cuda/include:$CPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY docker-entrypoint.sh /usr/local/bin/

RUN apt-get update && pip install \
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
	torch-geometric

RUN ln -s /usr/local/bin/docker-entrypoint.sh /

ENTRYPOINT ["docker-entrypoint.sh"]

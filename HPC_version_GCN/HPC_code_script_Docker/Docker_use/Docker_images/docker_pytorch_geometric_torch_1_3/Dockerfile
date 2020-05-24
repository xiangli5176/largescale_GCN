ARG  CODE_VERSION=1.3-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:${CODE_VERSION}

LABEL MAINTAINER="Xiang Li <xiangli13257@gmail.com>"

ENV CPATH=/usr/local/cuda/include:$CPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY docker-entrypoint.sh /usr/local/bin/

RUN apt-get update && apt-get install -y libmetis-dev && pip install \
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
        && pip uninstall networkx -y \
        && pip install -I networkx==2.3.0

RUN ln -s /usr/local/bin/docker-entrypoint.sh /

ENTRYPOINT ["docker-entrypoint.sh"]

# alternative effective way: for torch 1.2
# RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
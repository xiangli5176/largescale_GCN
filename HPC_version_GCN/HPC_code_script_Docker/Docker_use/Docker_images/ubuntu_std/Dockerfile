ARG  CODE_VERSION=latest
FROM ubuntu:${CODE_VERSION}

LABEL MAINTAINER="Xiang Li <xiangli13257@gmail.com>"

# ENV CPATH=/usr/local/cuda/include:$CPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY docker-entrypoint.sh /usr/local/bin/

RUN apt-get update

# RUN chmod 777 /usr/local/bin/docker-entrypoint.sh
RUN ln -s /usr/local/bin/docker-entrypoint.sh /

ENTRYPOINT ["docker-entrypoint.sh"]

# CMD ["bash"]

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        python3 python3-pip \
        r-base curl \
        libxml2-dev libcurl4-openssl-dev libssl-dev \
        libxt-dev build-essential && \
    apt-get clean

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

RUN Rscript -e "install.packages('data.table', repos = 'https://cloud.r-project.org')" && \
    Rscript -e "install.packages('BiocManager', repos = 'https://cloud.r-project.org')" && \
    Rscript -e "BiocManager::install('Biostrings', ask = FALSE, update = FALSE)" && \
    Rscript -e "install.packages('/app/R4RNA_2.0.9.tar.gz', repos = NULL, type = 'source')"

RUN chmod +x /app/run.sh

CMD ["/app/run.sh", "/app/examples"]

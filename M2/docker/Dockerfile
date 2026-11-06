# One image for every Hadoop role (namenode, resourcemanager, worker = datanode + nodemanager)
# and for the client that runs run.py. Native on Apple Silicon (arm64) and on x86 (amd64).
#   Hadoop 3.4.1 daemons on Java 11, Spark 4.0 (pip pyspark) driver/executors on Java 17,
#   Python 3.10 + requirements.txt, and /opt/kitenv (Python 3.9 + the kit's model library versions).
FROM ubuntu:22.04

ARG TARGETARCH
ARG HADOOP_VERSION=3.4.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-11-jdk-headless openjdk-17-jdk-headless \
        python3 python3-pip python3-venv python3-dev \
        build-essential gfortran libopenblas-dev \
        curl ca-certificates zip unzip procps netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture) /opt/java11 \
    && ln -s /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /opt/java17

# Hadoop: the generic release is pure Java and runs on arm64 too (only the optional native
# libraries are x86; Hadoop then logs "Unable to load native-hadoop library" and uses Java code).
RUN set -eux; \
    curl -fsSL "https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz" \
      | tar -xz -C /opt; \
    ln -s /opt/hadoop-${HADOOP_VERSION} /opt/hadoop; \
    rm -rf /opt/hadoop/share/doc /opt/hadoop/lib/native
ENV HADOOP_HOME=/opt/hadoop \
    HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop \
    PATH=/opt/hadoop/bin:/opt/hadoop/sbin:$PATH
COPY docker/hadoop-conf/ /opt/hadoop/etc/hadoop/
RUN echo 'export JAVA_HOME=/opt/java11' >> /opt/hadoop/etc/hadoop/hadoop-env.sh

# Pipeline Python (Spark driver and executors use the same interpreter)
COPY requirements.txt requirements-kit.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && PYSPARK_DIR=$(python3 -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))") \
    && curl -fsSL -o "$PYSPARK_DIR/jars/postgresql-42.7.4.jar" \
         https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar

# Kit model environment (tools/python/python.yaml of TPCx-AI v2.0.0) for --model-python.
# statsmodels 0.12.2 may have no wheel for this platform: build it against the pinned numpy.
RUN pip3 install --no-cache-dir uv \
    && uv venv --python 3.9 /opt/kitenv \
    && uv pip install --python /opt/kitenv/bin/python numpy==1.23.5 "scipy<1.12" "cython<3" "setuptools<70" wheel \
    && uv pip install --python /opt/kitenv/bin/python --no-build-isolation statsmodels==0.12.2 \
    && uv pip install --python /opt/kitenv/bin/python -r /tmp/requirements-kit.txt \
    && /opt/kitenv/bin/python -c "import statsmodels, xgboost, pandas; print(statsmodels.__version__, xgboost.__version__, pandas.__version__)"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV JAVA_HOME=/opt/java17 \
    SPARK_JAVA_HOME=/opt/java17 \
    PYSPARK_PYTHON=/usr/bin/python3
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]

FROM ubuntu:latest

RUN apt-get update && \
    apt-get install tree && \
    apt-get install -y bash git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 ./get-pip.py && \
    pip install pytest

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
RUN bash miniconda.sh -b -u -p ./miniconda3
RUN rm miniconda.sh
# support python=3.6 download
RUN /miniconda3/bin/conda config --add channels conda-forge
RUN /miniconda3/bin/conda init
RUN export PATH=/miniconda3/bin:$PATH

# for Github: https://stackoverflow.com/a/67390274
RUN export GIT_TRACE_PACKET=1
RUN export GIT_TRACE=1
RUN export GIT_CURL_VERBOSE=1

# RUN git clone https://github.com/swe-bench/pyvista__pyvista.git
# RUN git clone https://github.com/swe-bench/humaneval.git
# RUN git clone https://github.com/swe-bench/pydicom__pydicom.git
# RUN git clone https://github.com/swe-bench/pvlib__pvlib-python.git
# RUN git clone https://github.com/swe-bench/marshmallow-code__marshmallow.git
# RUN git clone https://github.com/swe-bench/pylint-dev__astroid.git
# RUN git clone https://github.com/swe-bench/mwaskom__seaborn.git
# RUN git clone https://github.com/swe-bench/sqlfluff__sqlfluff.git
# RUN git clone https://github.com/swe-bench/psf__requests.git
# RUN git clone https://github.com/swe-bench/pydata__xarray.git
# RUN git clone https://github.com/swe-bench/pylint-dev__pylint.git
# RUN git clone https://github.com/swe-bench/sympy__sympy.git
# RUN git clone https://github.com/swe-bench/astropy__astropy.git
# RUN git clone https://github.com/swe-bench/matplotlib__matplotlib.git
# RUN git clone https://github.com/swe-bench/pallets__flask.git
# RUN git clone https://github.com/swe-bench/pytest-dev__pytest.git
# RUN git clone https://github.com/swe-bench/django__django.git
# RUN git clone https://github.com/swe-bench/scikit-learn__scikit-learn.git
# RUN git clone https://github.com/ZiyueWang25/ziyuewang25__toyexamples.git


RUN git config --global user.email "intercode@pnlp.org"
RUN git config --global user.name "intercode"

WORKDIR /
FROM python:3

RUN apt-get update
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn

COPY classifier.py /classifier.py
COPY iterate_benchmarks.py /iterate_benchmarks.py

CMD python3 iterate_benchmarks.py

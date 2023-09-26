FROM rocm/tensorflow:rocm5.6-tf2.12-dev

WORKDIR /deps

RUN pip install pandas==1.4.2 && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

WORKDIR /models
RUN aws s3 cp s3://mutcomputex/models . --recursive --no-sign-request

COPY scripts/generate_norbelladine_predictions.py /scripts/generate_norbelladine_predictions.py
COPY MutComputeX /opt/MutComputeX/MutComputeX
ENV PYTHONPATH=/opt/MutComputeX
WORKDIR /scripts
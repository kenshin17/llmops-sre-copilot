FROM apache/airflow:2.9.3

# Airflow UID is 50000 by default in the base image
USER ${AIRFLOW_UID:-50000}

# Copy project code and DAGs with correct ownership
COPY --chown=${AIRFLOW_UID:-50000}:0 ../sre-copilot /opt/airflow/sre-copilot
COPY --chown=${AIRFLOW_UID:-50000}:0 ../sre-copilot/airflow/dags /opt/airflow/dags

# Install project dependencies as the airflow user
RUN pip install --no-cache-dir /opt/airflow/sre-copilot

# Ensure the package is importable
ENV PYTHONPATH="/opt/airflow/sre-copilot/src"

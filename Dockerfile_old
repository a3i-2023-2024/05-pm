# Specify the base image
FROM tensorflow/tensorflow:2.9.2

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip graphviz

# Install additional Python packages
RUN pip install jupyter==1.0.0 pandas==1.4.4 scikit-learn==1.1.2 matplotlib==3.5.3 ipympl==0.9.2 rise==5.7.1 jupyter-contrib-nbextensions==0.5.1 pydot==1.4.2 tensorflow-probability==0.17.0
RUN jupyter contrib nbextension install --system

# Make sure the contents of our repo are in /app
COPY . /app

# Specify working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]

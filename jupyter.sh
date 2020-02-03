jupyter tensorboard enable --user
jupyter notebook --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --allow-root --no-browser >> /data/logs/jupyter.log 2>&1 &
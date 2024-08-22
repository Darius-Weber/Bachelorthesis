# QP-GNN: Exploring the Power of Graph Neural Networks in Solving Convex Optimization Problems

## Main results 
```angular2html
See `run/main.sh` for the commands and hyperparameters. Simply uncommand the specific command and run it.
```

## Note:
```angular2html
This work builds on the recent study by Qian et al. [![arXiv](https://img.shields.io/badge/arXiv-2310.10603-b31b1b.svg)](https://arxiv.org/abs/2310.10603). Therefore, some parts of the code are modified versions of: https://github.com/chendiqian/IPM_MPNN
```

## Important packages:
```angular2html
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.1%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.17%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
pip install ml-collections
pip install wandb
```
## May not need all packages:
```angular2html
Package                   Version
------------------------- ----------------
absl-py                   2.1.0
aiohttp                   3.9.5
aiosignal                 1.3.1
anyio                     4.4.0
argon2-cffi               23.1.0
argon2-cffi-bindings      21.2.0
arrow                     1.3.0
asttokens                 2.4.1
async-lru                 2.0.4
async-timeout             4.0.3
attrs                     23.2.0
Babel                     2.15.0
beautifulsoup4            4.12.3
bleach                    6.1.0
certifi                   2024.6.2
cffi                      1.16.0
charset-normalizer        3.3.2
click                     8.1.7
comm                      0.2.2
contextlib2               21.6.0
cvxopt                    1.3.2
debugpy                   1.8.2
decorator                 5.1.1
defusedxml                0.7.1
docker-pycreds            0.4.0
exceptiongroup            1.2.1
executing                 2.0.1
fastjsonschema            2.20.0
filelock                  3.15.4
fqdn                      1.5.1
frozenlist                1.4.1
fsspec                    2024.6.1
gitdb                     4.0.11
GitPython                 3.1.43
h11                       0.14.0
httpcore                  1.0.5
httpx                     0.27.0
idna                      3.7
ipykernel                 6.29.5
ipython                   8.26.0
isoduration               20.11.0
jedi                      0.19.1
Jinja2                    3.1.4
joblib                    1.4.2
json5                     0.9.25
jsonpointer               3.0.0
jsonschema                4.22.0
jsonschema-specifications 2023.12.1
jupyter_client            8.6.2
jupyter_core              5.7.2
jupyter-events            0.10.0
jupyter-lsp               2.2.5
jupyter_server            2.14.1
jupyter_server_terminals  0.5.3
jupyterlab                4.2.3
jupyterlab_pygments       0.3.0
jupyterlab_server         2.27.2
MarkupSafe                2.1.5
matplotlib-inline         0.1.7
mistune                   3.0.2
ml-collections            0.1.1
mpmath                    1.3.0
multidict                 6.0.5
nbclient                  0.10.0
nbconvert                 7.16.4
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.3
notebook                  7.2.1
notebook_shim             0.2.4
numpy                     2.0.0
nvidia-cublas-cu12        12.1.3.1
nvidia-cuda-cupti-cu12    12.1.105
nvidia-cuda-nvrtc-cu12    12.1.105
nvidia-cuda-runtime-cu12  12.1.105
nvidia-cudnn-cu12         8.9.2.26
nvidia-cufft-cu12         11.0.2.54
nvidia-curand-cu12        10.3.2.106
nvidia-cusolver-cu12      11.4.5.107
nvidia-cusparse-cu12      12.1.0.106
nvidia-nccl-cu12          2.20.5
nvidia-nvjitlink-cu12     12.5.82
nvidia-nvtx-cu12          12.1.105
overrides                 7.7.0
packaging                 24.1
pandocfilters             1.5.1
parso                     0.8.4
pexpect                   4.9.0
pip                       22.0.2
platformdirs              4.2.2
prometheus_client         0.20.0
prompt_toolkit            3.0.47
protobuf                  5.27.2
psutil                    6.0.0
ptyprocess                0.7.0
pure-eval                 0.2.2
pycparser                 2.22
Pygments                  2.18.0
pyparsing                 3.1.2
python-dateutil           2.9.0.post0
python-json-logger        2.0.7
PyYAML                    6.0.1
pyzmq                     26.0.3
referencing               0.35.1
requests                  2.32.3
rfc3339-validator         0.1.4
rfc3986-validator         0.1.1
rpds-py                   0.18.1
scikit-learn              1.5.0
scipy                     1.14.0
Send2Trash                1.8.3
sentry-sdk                2.7.1
setproctitle              1.3.3
setuptools                59.6.0
six                       1.16.0
smmap                     5.0.1
sniffio                   1.3.1
soupsieve                 2.5
stack-data                0.6.3
sympy                     1.12.1
terminado                 0.18.1
threadpoolctl             3.5.0
tinycss2                  1.3.0
tomli                     2.0.1
torch                     2.3.1
torch_geometric           2.5.3
torch_scatter             2.1.2+pt23cu121
torch_sparse              0.6.18+pt23cu121
tornado                   6.4.1
tqdm                      4.66.4
traitlets                 5.14.3
triton                    2.3.1
types-python-dateutil     2.9.0.20240316
typing_extensions         4.12.2
uri-template              1.3.0
urllib3                   2.2.2
wandb                     0.17.3
wcwidth                   0.2.13
webcolors                 24.6.0
webencodings              0.5.1
websocket-client          1.8.0
yarl                      1.9.4
```
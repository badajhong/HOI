#!/usr/bin/env bash
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if ! command -v sudo &> /dev/null; then
  # in docker build sudo isn't avaiable, but its ok
  echo "Warning: sudo could not be found, you may need to run this script with sudo"
  function sudo { "$@"; }
  export -f sudo
fi

# Use CONDA_ENV_NAME if provided, otherwise default to "hssim"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hssim}
echo "conda environment name is set to: $CONDA_ENV_NAME"

# Isaac Sim prompts for EULA acceptance when this is unset. Export it here so
# setup also works without an interactive stdin (for example, under nohup).
export OMNI_KIT_ACCEPT_EULA=1

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$CONDA_ENV_NAME
CONDA_BIN=$CONDA_ROOT/bin/conda
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_$CONDA_ENV_NAME
echo "SENTINEL_FILE: $SENTINEL_FILE"

# A reset may delete Miniconda while its environment is still active in the
# parent shell. Do not let that stale activation state affect the new Conda.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
export CONDA_SHLVL=0

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -x $CONDA_BIN ]]; then
    # Recover from partial installs where CONDA_ROOT exists but conda is missing.
    rm -rf $CONDA_ROOT
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_BIN tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if [[ ! -f $CONDA_ROOT/bin/mamba ]]; then
      $CONDA_BIN install -y mamba -c conda-forge -n base
    fi
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $CONDA_ENV_NAME python=3.11 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $CONDA_ENV_NAME

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install -U \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  pip install pyperclip
  # Then install isaacsim from NVIDIA index only
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  if [[ ! -d $WORKSPACE_DIR/IsaacLab ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 $WORKSPACE_DIR/IsaacLab
  fi

  # Origin
  # sudo apt install -y cmake build-essential
  # New
  conda install -c conda-forge -y cmake compilers
  
  cd $WORKSPACE_DIR/IsaacLab
  # setuptools 81 removes pkg_resoures, a dep needs that
  # see https://github.com/isaac-sim/IsaacLab/pull/4585
  pip install 'setuptools<81'
  echo 'setuptools<81' > build-constraints.txt
  export PIP_BUILD_CONSTRAINT="$(realpath build-constraints.txt)"
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  ./isaaclab.sh --install
  unset PIP_BUILD_CONSTRAINT

 # Install Holosoma
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Holosoma and IsaacLab install broad/new dependencies after Isaac Sim. Restore
  # the versions required by Isaac Sim 5.1 and keep RL packages compatible with
  # its PyTorch 2.7 runtime.
  pip install --upgrade \
    'click==8.1.7' \
    'huggingface-hub==0.36.0' \
    'ipython==8.31.0' \
    'jupyter-client==8.6.3' \
    'jupyterlab==4.3.4' \
    'notebook==7.3.2' \
    'numpy==1.26.0' \
    'onnx==1.18.0' \
    'packaging==23.0' \
    'psutil==5.9.8' \
    'stable-baselines3==2.7.0' \
    'transformers==4.57.6' \
    'typeguard==4.4.1' \
    'typing_extensions==4.12.2' \
    'wandb==0.22.0' \
    'wheel==0.45.1'

  # Tyro 1.0 provides the CLI markers used by Holosoma. Its declared
  # typing_extensions floor is newer than Isaac Sim's exact pin, but the used
  # API is compatible with 4.12.2 (covered by test_tyro_cli.py).
  pip install --no-deps 'tyro==1.0.0'

  # Some IsaacLab extras can remove torchaudio while resolving their own
  # dependencies, so restore the CUDA-matched build after all other installs.
  pip install --upgrade \
    'torchaudio==2.7.0' \
    --index-url https://download.pytorch.org/whl/cu128

  # Refuse unexpected dependency conflicts. Ignore only Tyro's conservative
  # metadata floor described above.
  PIP_CHECK_OUTPUT=$(pip check 2>&1 || true)
  UNEXPECTED_PIP_CONFLICTS=$(printf '%s\n' "$PIP_CHECK_OUTPUT" | grep -v '^tyro 1.0.0 has requirement typing-extensions>=4.13.0' || true)
  if [[ -n "$UNEXPECTED_PIP_CONFLICTS" && "$UNEXPECTED_PIP_CONFLICTS" != "No broken requirements found." ]]; then
    printf '%s\n' "$PIP_CHECK_OUTPUT"
    exit 1
  fi
  touch $SENTINEL_FILE
fi

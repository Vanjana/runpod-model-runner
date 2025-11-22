# install Linux packages
apt update
apt install -y vim

# copy ssh key
cp -ar /workspace/.ssh ~/

if [ ! -d /models ]; then
  mkdir /models
fi

if [ ! -d /workspace/app ]; then
  git clone git@github.com:Vanjana/runpod-model-runner.git /workspace/app
fi

if [ ! -d /workspace/venv ]; then
  python -m venv /workspace/venv
  source /workspace/venv/bin/activate

  cd /workspace/app
  pip install -r requirements.txt
fi

pip install --upgrade pip

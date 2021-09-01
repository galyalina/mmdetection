#Get conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -bfp /usr/local
conda update conda -y -q
source /usr/local/etc/profile.d/conda.sh
conda init
conda install -n root _license -y -q
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y

# install the latest mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
pip install ipykernel

# install mmdetection
pip install -r requirements/build.txt
pip install -v -e .


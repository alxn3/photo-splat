# PhotoSplat

## Getting Started

Make sure that conda and cuda is installed and then run the following commands to set up the environment:
```sh
# Create a conda environment
conda create -y -n splat python=3.11
conda activate splat
conda install -c conda-forge gcc gxx ninja

# Install NoPoSplat and its dependencies
git clone https://github.com/cvg/NoPoSplat  

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r NoPoSplat/requirements.txt

# Needed due to breaking changes
pip uninstall moviepy
pip install moviepy==1.0.3

cd NoPoSplat/src/model/encoder/backbone/croco/curope/
python setup.py build_ext --inplace
cd -

wget -P pretrained https://huggingface.co/botaoye/NoPoSplat/resolve/main/mixRe10kDl3dv_512x512.ckpt 

cp predict.yaml NoPoSplat/config/   
```

# PhotoSplat

## Getting Started

Make sure that conda and cuda is installed and then run the following commands to set up the environment:
```sh
# Create a conda environment
conda create -y -n splat python=3.11
conda activate splat
conda install -y -c conda-forge gcc=12 gxx ninja cccl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # Replace with your CUDA version.

pip install ninja -vvv git+https://github.com/facebookresearch/pytorch3d.git 

# Install FLARE and its dependencies
git clone https://github.com/ant-research/FLARE.git
pip install -r FLARE/requirements.txt

conda uninstall ffmpeg 
conda install -c conda-forge ffmpeg -y

# Build CUDA acceralated RoPE
cd curope
python setup.py build_ext --inplace
cd -
cp -r curope FLARE/dust3r/croco/models/

# Fix deprecated warnings
find FLARE/ -type f -exec sed -i "s/torch.cuda.amp.autocast(/torch.amp.autocast('cuda', /g" {} \;  

# Fix circular dependency
find FLARE/dust3r/dust3r/ -type f -exec sed -i "s/from dust3r.utils.image import imread_cv2/# from dust3r.utils.image import imread_cv2/g" {} \;

# Download model checkpoints
wget -P pretrained https://huggingface.co/AntResearch/FLARE/resolve/main/geometry_pose.pth
wget -P pretrained https://huggingface.co/zhang3z/FLARE_NVS/resolve/main/NVS.pth

pip install plyfile

pip install -r requirements.txt
```

## Running

To run the FLARE model, please use the `run.py` script to do so. `python run.py --help` for more information.

The included Jupyter notebook has examples on how to use the outputs to edit photo parameters.

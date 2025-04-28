# PhotoSplat

## Getting Started

Make sure that conda and cuda is installed and then run the following commands to set up the environment:
```sh
# Create a conda environment
conda create -y -n splat python=3.11
conda activate splat
conda install -y -c conda-forge gcc gxx ninja cccl

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # Replace with your CUDA version.

# Install FLARE and its dependencies
git clone https://github.com/ant-research/FLARE.git
pip install -r FLARE/requirements.txt

pip install ninja -vvv git+https://github.com/facebookresearch/pytorch3d.git 

# Build CUDA acceralated RoPE
cp -r curope FLARE/dust3r/croco/models/
cd FLARE/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd -

# Download model checkpoints
wget -P pretrained https://huggingface.co/AntResearch/FLARE/resolve/main/geometry_pose.pth
wget -P pretrained https://huggingface.co/zhang3z/FLARE_NVS/resolve/main/NVS.pth

pip install plyfile
```

## Running

To train a gaussian splat, run the following:
```sh
torchrun --nproc_per_node=1 FLARE/run_pose_pointcloud.py \
    --test_dataset "1 @ CustomDataset(split='train', ROOT='<path/to/input/images>', resolution=(<img_width>,<img_height>), seed=1, num_views=2, gt_num_image=0, aug_portrait_or_landscape=True, sequential_input=False, wpose=False)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf), wpose=False)" \
    --pretrained ./pretrained/geometry_pose.pth \
    --test_criterion "MeshOutput(sam=False)" --output_dir "log/" --amp 1 --seed 1 --num_workers 0
```
- `img_width` and `img_height` should match the aspect ratio of input images and be a multiple of `64`.

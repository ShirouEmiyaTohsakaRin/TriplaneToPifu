set -ex

mkdir -p checkpoints
cd checkpoints
wget "https://drive.google.com/uc?export=download&id=1zEmVXG2VHy0MMzngcRshB4D8Sr_oLHsm" -O net_G
wget "https://drive.google.com/uc?export=download&id=1V83B6GDIjYMfHdpg-KcCSAPgHxpafHgd" -O net_C
cd ..
echo "Downloaded original PIFu weights. Initialize hybrid shape training with --init_pifu_checkpoint checkpoints/net_G."
echo "net_C is an optional RGB initializer; these files are not trained triplane weights."

TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch run.py --dataset concode
#GLOO_SOCKET_IFNAME="enp129s0f0"
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NCCL_IB_DISABLE=1 NCCL_IB_GID_INDEX=3 OMP_NUM_THREADS=4 NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME torchrun --nnodes 2 --node_rank 1 --nproc_per_node 6 --master_addr="162.105.88.154" --master_port=9925 run.py
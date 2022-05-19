python3 eval_mesh.py \
	--mesh /gpfs/data/gpfs0/egor.burkov/Projects/NeuS/logs-paper/h3ds/100_rank100_smallLR/frontal/f7e930d8a9ff2091-1-0000/meshes/0_00025000.ply \
	--gt_mesh /gpfs/data/gpfs0/egor.burkov/Datasets/H3DS_preprocessed/f7e930d8a9ff2091/mesh.ply \
	--scene_id f7e930d8a9ff2091 \
	--out_dir metrics/outputs-2022-04-04_test


# compute chamfer distance
for model_name in 100_rank50_smallLR_initSimilar0; do
for view in frontal left right; do
for scene_id in 1b2a8613401e42a8 3b5a2eb92a501d54 444ea0dc5e85ee0b 5ae021f2805c0854 5cd49557ea450c89 609cc60fd416e187 7dd427509fe84baa 868765907f66fd85 e98bae39fad2244e f7e930d8a9ff2091; do

exp_dir=`echo logs-paper/h3ds/${model_name}/${view}/${scene_id}*`

# for m in ../${exp_dir}/meshes/*; do
# python3 apply_transform_to_mesh_from_h3dnet.py --mesh $m --scene_id ${scene_id} --output $m
# done

srun -c 2 -t 0:10:0 -p cpu,mem,gpu,gpu_devel,htc,gpu_a100 python3 eval_mesh.py \
--mesh `ls -1d ${exp_dir}/meshes/* | tail -n 1` \
--scene_id ${scene_id} > ${exp_dir}/eval_mesh_${scene_id}.txt 2>&1 &
srun -c 2 -t 0:10:0 -p cpu,mem,gpu,gpu_devel,htc,gpu_a100 python3 eval_mesh.py \
--mesh `ls -1d ${exp_dir}/meshes/* | head -n 1` \
--scene_id ${scene_id} > ${exp_dir}/eval_mesh_scenewiseOnly_${scene_id}.txt 2>&1 &

done
done
done

# print chamfer distance
for model_name in 100_rank50_smallLR_initSimilar0; do
for ft_mode in scenewise full; do
for view in frontal left right; do
for scene_id in 1b2a8613401e42a8 3b5a2eb92a501d54 444ea0dc5e85ee0b 5ae021f2805c0854 5cd49557ea450c89 609cc60fd416e187 7dd427509fe84baa 868765907f66fd85 e98bae39fad2244e f7e930d8a9ff2091; do

exp_dir=`echo logs-paper/h3ds/${model_name}/${view}/${scene_id}*`
if [ ${ft_mode} = "scenewise" ]; then
# printf "$(cat ${exp_dir}/eval_mesh_scenewiseOnly_${scene_id}.txt | grep sphere | cut -c 39-53) "
printf "$(cat ${exp_dir}/eval_mesh_scenewiseOnly_${scene_id}.txt | grep full | cut -c 37-51) "
else
# printf "$(cat ${exp_dir}/eval_mesh_${scene_id}.txt | grep sphere | cut -c 39-53) "
printf "$(cat ${exp_dir}/eval_mesh_${scene_id}.txt | grep full | cut -c 37-51) "
fi

done
echo
done
done
done



declare -A view_idx
view_idx['1b2a8613401e42a8','frontal']=0
view_idx['1b2a8613401e42a8','left']=18
view_idx['1b2a8613401e42a8','right']=65
view_idx['3b5a2eb92a501d54','frontal']=1
view_idx['3b5a2eb92a501d54','left']=15
view_idx['3b5a2eb92a501d54','right']=62
view_idx['444ea0dc5e85ee0b','frontal']=0
view_idx['444ea0dc5e85ee0b','left']=12
view_idx['444ea0dc5e85ee0b','right']=59
view_idx['5ae021f2805c0854','frontal']=1
view_idx['5ae021f2805c0854','left']=7
view_idx['5ae021f2805c0854','right']=54
view_idx['5cd49557ea450c89','frontal']=0
view_idx['5cd49557ea450c89','left']=14
view_idx['5cd49557ea450c89','right']=63
view_idx['609cc60fd416e187','frontal']=0
view_idx['609cc60fd416e187','left']=14
view_idx['609cc60fd416e187','right']=61
view_idx['7dd427509fe84baa','frontal']=1
view_idx['7dd427509fe84baa','left']=15
view_idx['7dd427509fe84baa','right']=64
view_idx['868765907f66fd85','frontal']=0
view_idx['868765907f66fd85','left']=15
view_idx['868765907f66fd85','right']=62
view_idx['e98bae39fad2244e','frontal']=0
view_idx['e98bae39fad2244e','left']=13
view_idx['e98bae39fad2244e','right']=61
view_idx['f7e930d8a9ff2091','frontal']=1
view_idx['f7e930d8a9ff2091','left']=14
view_idx['f7e930d8a9ff2091','right']=61

# transform h3d net single-view output meshes
for view in frontal left right; do
for scene_id in 1b2a8613401e42a8 3b5a2eb92a501d54 444ea0dc5e85ee0b 5ae021f2805c0854 5cd49557ea450c89 609cc60fd416e187 7dd427509fe84baa 868765907f66fd85 e98bae39fad2244e f7e930d8a9ff2091; do
src="h3d_net_single_view/original/${scene_id}/${view_idx[${scene_id},${view}]}/mesh.ply"
dst="h3d_net_single_view/multineus-transform-applied/${scene_id}/h3dnet_single_${view}.ply"
mkdir -p $(dirname ${dst})
python3 apply_transform_to_mesh_from_h3dnet.py --mesh ${src} --scene_id ${scene_id} --output ${dst}
done
done

# transform h3d net multi-view output meshes
for num_views in 3 4; do
for scene_id in 1b2a8613401e42a8 3b5a2eb92a501d54 444ea0dc5e85ee0b 5ae021f2805c0854 5cd49557ea450c89 609cc60fd416e187 7dd427509fe84baa 868765907f66fd85 e98bae39fad2244e f7e930d8a9ff2091; do
src="h3d_net_multi_view/original/${num_views}/${scene_id}_${num_views}.ply"
dst="h3d_net_multi_view/multineus-transform-applied/${scene_id}/h3dnet_${num_views}.ply"
mkdir -p $(dirname ${dst})
python3 apply_transform_to_mesh_from_h3dnet.py --mesh ${src} --scene_id ${scene_id} --output ${dst}
done
done


# regenerate meshes
PORT=26000
for model_name in 100_rank10_smallLR; do
for view in frontal left right; do
for scene_id in 1b2a8613401e42a8 3b5a2eb92a501d54 444ea0dc5e85ee0b 5ae021f2805c0854 5cd49557ea450c89 609cc60fd416e187 7dd427509fe84baa 868765907f66fd85 e98bae39fad2244e f7e930d8a9ff2091; do

DIR=$(echo logs-paper/h3ds/${model_name}/${view}/${scene_id}*)

# first mesh
srun -c 2 --gres=gpu:1 -t 0:2:30 -p gpu,gpu_devel,res,htc,gpu_a100 \
torchrun \
--rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=1 exp_runner.py \
--mode validate_mesh \
--checkpoint_path $DIR/checkpoints/ckpt_0015000.pth \
--extra_config_args "general { base_exp_dir = \"${DIR}\" }, dataset { images_to_pick = [[0, \"default\"]] }" > /dev/null 2>&1 &
let "PORT+=1"

# last mesh
srun -c 2 --gres=gpu:1 -t 0:2:30 -p gpu,gpu_devel,res,htc,gpu_a100 \
torchrun \
--rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=1 exp_runner.py \
--mode validate_mesh \
--checkpoint_path $DIR/checkpoints/ckpt_0042500.pth \
--extra_config_args "general { base_exp_dir = \"${DIR}\" }, dataset { images_to_pick = [[0, \"default\"]] }" > /dev/null 2>&1 &
let "PORT+=1"

done
done
done

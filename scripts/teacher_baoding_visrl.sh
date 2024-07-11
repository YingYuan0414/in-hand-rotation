GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=ball task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=64 train.params.config.minibatch_size=1024 \
train.params.config.central_value_config.minibatch_size=1024 \
task.env.observationType=partial_stack_baoding task.env.legacy_obs=False \
task.env.ablation_mode=multi-modality-plus task.env.pc_mode=label \
experiment=baoding-visualrl \
train.params.config.user_prefix=baoding-visualrl \
wandb_activate=True \
${EXTRA_ARGS}
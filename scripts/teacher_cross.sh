GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train.py headless=True \
task.env.objSet=cross task=AllegroArmMOAR task.env.axis=z \
task.env.numEnvs=8192 train.params.config.minibatch_size=16384 \
train.params.config.central_value_config.minibatch_size=16384 \
task.env.observationType=full_stack task.env.legacy_obs=True \
task.env.ablation_mode=no-pc experiment=wheel-wrench \
train.params.config.user_prefix=wheel-wrench wandb_activate=True \
${EXTRA_ARGS}
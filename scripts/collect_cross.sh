GPUS=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python ./isaacgymenvs/train_distillation.py headless=True \
distill.teacher_data_dir=demonstration-cross \
task.env.legacy_obs=False distill.bc_training=collect \
task.env.objSet=cross task.env.is_distillation=True \
train.params.config.user_prefix=bc-cross-collect task=AllegroArmMOAR \
task.env.numEnvs=64 train.params.config.minibatch_size=1024 \
experiment=bc-cross-collect wandb_activate=False \
task.env.observationType=full_stack \
train.params.config.central_value_config.minibatch_size=1024 \
distill.worker_id=0 distill.ablation_mode=multi-modality-plus \
task.env.ablation_mode=multi-modality-plus \
${EXTRA_ARGS}
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# from distillation.rl_pytorch.ppo import ActorCritic, get_encoder
from distillation.rl_pytorch.rl_pytorch.distill.distill_collect import DistillCollector
from distillation.rl_pytorch.rl_pytorch.distill.distill_bc_warmup import DistillWarmUpTrainer


def process_distill_trainer(env, cfg_train, teacher_logdir, student_logdir, teacher_params, student_params, enable_wandb, bc_warmup,
        teacher_data_dir, worker_id, bc_training=False, warmup_mode=None, player=None, batch_size=8192, ablation_mode=None):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]
    teacher_resume = cfg_train["teacher_resume"]

    if bc_training == "warmup":
        distill_trainer = DistillWarmUpTrainer(
            teacher_params=teacher_params,
            student_params=student_params,
            vec_env=env,
            num_transitions_per_env=learn_cfg["nsteps"],
            num_learning_epochs=learn_cfg["noptepochs"],
            num_mini_batches=learn_cfg["nminibatches"],
            clip_param=learn_cfg["cliprange"],
            gamma=learn_cfg["gamma"],
            lam=learn_cfg["lam"],
            init_noise_std=learn_cfg.get("init_noise_std", 0.05),
            surrogate_loss_coef=learn_cfg.get(
                "surrogate_loss_coef", 1.0),
            value_loss_coef=learn_cfg.get(
                "value_loss_coef", 2.0),
            bc_loss_coef=learn_cfg.get(
                "bc_loss_coef", 1.0),
            entropy_coef=learn_cfg["ent_coef"],
            use_l1=learn_cfg["use_l1"],
            learning_rate=learn_cfg["optim_stepsize"],
            weight_decay=learn_cfg["weight_decay"],
            max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
            use_clipped_value_loss=learn_cfg.get(
                "use_clipped_value_loss", True),
            schedule=learn_cfg.get("schedule", "fixed"),
            desired_kl=learn_cfg.get("desired_kl", None),
            device=learn_cfg.get('device', 'cuda:0'),
            sampler=learn_cfg.get("sampler", 'sequential'),
            teacher_log_dir=teacher_logdir,
            student_log_dir=student_logdir,
            is_testing=is_testing,
            print_log=learn_cfg["print_log"],
            apply_reset=False,
            teacher_resume=teacher_resume,
            vidlogdir=cfg_train.vidlogdir,
            vid_log_step=cfg_train.vid_log_step,
            log_video=cfg_train.log_video,
            enable_wandb=enable_wandb,
            bc_warmup=bc_warmup,
            teacher_data_dir=teacher_data_dir,
            worker_id=worker_id,
            warmup_mode=warmup_mode,
            ablation_mode=ablation_mode
        )
    elif bc_training == "collect":
        distill_trainer = DistillCollector(
            teacher_params=teacher_params,
            student_params=student_params,
            vec_env=env,
            num_transitions_per_env=learn_cfg["nsteps"],
            num_learning_epochs=learn_cfg["noptepochs"],
            num_mini_batches=learn_cfg["nminibatches"],
            clip_param=learn_cfg["cliprange"],
            gamma=learn_cfg["gamma"],
            lam=learn_cfg["lam"],
            init_noise_std=learn_cfg.get("init_noise_std", 0.05),
            surrogate_loss_coef=learn_cfg.get(
                "surrogate_loss_coef", 1.0),
            value_loss_coef=learn_cfg.get(
                "value_loss_coef", 2.0),
            bc_loss_coef=learn_cfg.get(
                "bc_loss_coef", 1.0),
            entropy_coef=learn_cfg["ent_coef"],
            use_l1=learn_cfg["use_l1"],
            learning_rate=learn_cfg["optim_stepsize"],
            max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
            use_clipped_value_loss=learn_cfg.get(
                "use_clipped_value_loss", True),
            schedule=learn_cfg.get("schedule", "fixed"),
            desired_kl=learn_cfg.get("desired_kl", None),
            device=learn_cfg.get('device', 'cuda:0'),
            sampler=learn_cfg.get("sampler", 'sequential'),
            teacher_log_dir=teacher_logdir,
            student_log_dir=student_logdir,
            is_testing=is_testing,
            print_log=learn_cfg["print_log"],
            apply_reset=False,
            teacher_resume=teacher_resume,
            vidlogdir=cfg_train.vidlogdir,
            vid_log_step=cfg_train.vid_log_step,
            log_video=cfg_train.log_video,
            enable_wandb=enable_wandb,
            bc_warmup=bc_warmup,
            teacher_data_dir=teacher_data_dir,
            worker_id=worker_id,
            warmup_mode=warmup_mode,
            batch_size=batch_size
        )
    else:
        raise NotImplementedError

    return distill_trainer

from ..diffusion.gaussian import get_named_beta_schedule, LossType, ModelMeanType, ModelVarType
from ..diffusion.diffproc import SpacedDiffusion, space_timesteps
from ..unet import UNetModel

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=None,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        dims=3,
        num_groups=32,
        in_channels=8,
    )

def register_diffusion_arguments(config_diffusion):
    arguments = diffusion_defaults()
    for key, val in config_diffusion.items():
        if key in arguments:
            arguments[key] = val
    return arguments

def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        dims=3,
        num_groups=32,
        in_channels=8,
        out_channels=0,  # automatically determine if 0
        bottleneck_attention=True,
        resample_2d=True,
        additive_skips=False,
        predict_xstart=False,
        use_conditional_model=None
    )
    return res

def register_model_arguments(config_model):
    arguments = model_defaults()
    for key, val in config_model.items():
        if key in arguments:
            arguments[key] = val
    return arguments

def model_and_diffusion_defaults(config_diff: dict,
                                 config_model: dict,
                                 config_common: dict):
    conf = register_diffusion_arguments(config_diff)
    conf.update(register_model_arguments(config_model))
    for key in config_common:
        if key in conf:
            conf[key] = config_common[key]
    return conf

def create_model_and_diffusion(
    image_size,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    dims,
    num_groups,
    in_channels,
    out_channels,
    bottleneck_attention,
    resample_2d,
    additive_skips,
    use_conditional_model
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims,
        num_groups=num_groups,
        in_channels=in_channels,
        out_channels=out_channels,
        bottleneck_attention=bottleneck_attention,
        resample_2d=resample_2d,
        additive_skips=additive_skips,
        use_conditional_model=use_conditional_model,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    use_conditional_model,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=True,
    use_new_attention_order=False,
    num_groups=32,
    dims=2,
    in_channels=1,
    out_channels=0,  # automatically determine if 0
    bottleneck_attention=True,
    resample_2d=True,
    additive_skips=False
):
    if not channel_mult:
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 2, 2, 4, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"[MODEL] Unsupported image size: {image_size}")
    else:
        if isinstance(channel_mult, str):
            from ast import literal_eval
            channel_mult = literal_eval(channel_mult)
        elif isinstance(channel_mult, tuple) or isinstance(channel_mult, list):  # do nothing
            pass
        else:
            raise ValueError(f"[MODEL] Value for {channel_mult=} not supported")

    attention_ds = []
    if attention_resolutions:
        if isinstance(attention_resolutions, str):
            attention_resolutions = attention_resolutions.split(",")
        for res in attention_resolutions:
            attention_ds.append(image_size // int(res))
    if out_channels == 0:
        out_channels = (2*in_channels if learn_sigma else in_channels)

    return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=out_channels * (1 if not learn_sigma else 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=None,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            dims=dims,
            num_groups=num_groups,
            bottleneck_attention=bottleneck_attention,
            additive_skips=additive_skips,
            resample_2d=resample_2d,
            use_conditional_model=use_conditional_model,
    )

def create_gaussian_diffusion(
    steps=None,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    mode='default',
    *args,
    **kwargs
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        mode=mode
    )
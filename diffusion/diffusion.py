from .base import *
from dataclasses import dataclass
from typing import List, Set, Tuple, Union, Optional, Sequence


def space_timesteps(num_timesteps: int, section_counts: Union[str, List[int]]) -> Set[int]:
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}")
        
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
            
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
            
        all_steps += taken_steps
        start_idx += size
        
    return set(all_steps)


@dataclass
class SpacedDiffusionBeatGansConfig(GaussianDiffusionBeatGansConfig):
    use_timesteps: Optional[Sequence[int]] = None

    def make_sampler(self):
        return SpacedDiffusionBeatGans(self)


class SpacedDiffusionBeatGans(GaussianDiffusionBeatGans):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param conf: Configuration with use_timesteps - a collection (sequence or set) 
                of timesteps from the original diffusion process to retain.
    """
    def __init__(self, conf: SpacedDiffusionBeatGansConfig):
        self.conf = conf
        self.use_timesteps = set(conf.use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(conf.betas)

        # Create base diffusion to access alphas_cumprod
        base_conf = GaussianDiffusionBeatGansConfig(**{
            k: v for k, v in vars(conf).items() 
            if k != 'use_timesteps'
        })
        base_diffusion = GaussianDiffusionBeatGans(base_conf)
        
        last_alpha_cumprod = 1.0
        new_betas = []
        
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
                
        conf.betas = np.array(new_betas)
        super().__init__(conf)

    def p_mean_variance(self, model: Model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model: Model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model: Model) -> '_WrappedModel':
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, 
            self.timestep_map, 
            self.rescale_timesteps,
            self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is handled by the wrapped model
        return t


class _WrappedModel:
    """
    Converting the supplied t's to the old t's scales.
    """
    def __init__(self, model: Model, timestep_map: List[int], 
                rescale_timesteps: bool, original_num_steps: int):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, t, t_cond=None, **kwargs):
        """
        Args:
            x: Input tensor
            t: Timesteps with different ranges that need to be converted to original timesteps
            t_cond: Optional conditioning timesteps (same as t but can have different values)
        """
        map_tensor = th.tensor(
            self.timestep_map,
            device=t.device,
            dtype=t.dtype
        )

        def map_timesteps(time_steps):
            new_ts = map_tensor[time_steps]
            if self.rescale_timesteps:
                new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            return new_ts

        mapped_t = map_timesteps(t)
        mapped_t_cond = map_timesteps(t_cond) if t_cond is not None else None

        return self.model(x=x, t=mapped_t, t_cond=mapped_t_cond, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"Neither _WrappedModel nor inner model has attribute '{name}'")

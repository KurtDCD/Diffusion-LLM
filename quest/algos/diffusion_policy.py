import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.baseline_modules.diffusion_modules import ConditionalUnet1D
from diffusers.training_utils import EMAModel
from quest.algos.base import ChunkPolicy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import quest.utils.metaworld_utils as mu

class DiffusionPolicy(ChunkPolicy):
    def __init__(
            self, 
            diffusion_model,
            **kwargs
            ):
        super().__init__(**kwargs)
        
        self.diffusion_model = diffusion_model.to(self.device)

    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)
        loss = self.diffusion_model(cond, data["actions"])
        info = {
            'loss': loss.item(),
        }
        return loss, info
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        cond = self.get_cond(data)
        actions = self.diffusion_model.get_action(cond, data['task_id'], data['obs']['robot_states'])
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def get_cond(self, data):
        obs_emb = self.obs_encode(data)
        obs_emb = obs_emb.reshape(obs_emb.shape[0], -1)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond
    

class DiffusionModel(nn.Module):
    def __init__(self, 
                 noise_scheduler,
                 action_dim,
                 global_cond_dim,
                 diffusion_step_emb_dim,
                 down_dims,
                 ema_power,
                 skill_block_size,
                 diffusion_inf_steps,
                 device,
                 n_guide_steps=2,
                 guidance_scale=0.1,
                 guide_fn=None,
                 verbose=False,
                 ):
        super().__init__()
        self.device = device
        net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_emb_dim,
            down_dims=down_dims,
        ).to(self.device)
        self.ema = EMAModel(
            parameters=net.parameters(),
            decay=ema_power)
        self.net = net
        self.noise_scheduler = noise_scheduler
        self.action_dim = action_dim
        self.skill_block_size = skill_block_size
        self.diffusion_inf_steps = diffusion_inf_steps
        self.n_guide_steps = n_guide_steps
        self.guidance_scale = guidance_scale
        self.guide_fn = guide_fn
        self.verbose = verbose

    def forward(self, cond, actions):
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (cond.shape[0],), device=self.device
        ).long()
        noise = torch.randn(actions.shape, device=self.device)
        # add noise to the clean actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, noise, timesteps)
        # predict the noise residual
        noise_pred = self.net(
            noisy_actions, timesteps, global_cond=cond)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def get_action(self, cond, task_id, robot_states):
        nets = self.net
        noisy_action = torch.randn(
            (cond.shape[0], self.skill_block_size, self.action_dim), device=self.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.diffusion_inf_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets(
                sample=naction, 
                timestep=k,
                global_cond=cond
            )
            original_noise_pred = noise_pred

            # Apply guidance if provided
            if self.guide_fn is not None and k >= self.noise_scheduler.timesteps[-2]:
                # Get current alpha values from scheduler
                alpha_t_bar = self.noise_scheduler.alphas_cumprod[k]
                
                for _ in range(self.n_guide_steps):
                    # Convert predicted noise to x0
                    # x0 = (xt - sqrt(1-αt) * εt) / sqrt(αt)
                    x0_pred = (naction - torch.sqrt(1 - alpha_t_bar) * noise_pred) / torch.sqrt(alpha_t_bar)
                    
                    # Get gradient from guidance function
                    with torch.enable_grad():
                        x0_pred.requires_grad_(True)
                        loss = self.guide_fn(task_id, x0_pred, robot_states)
                        grad = torch.autograd.grad(loss, x0_pred)[0]
                        x0_pred.requires_grad_(False)
                    
                    # Scale gradient based on loss magnitude
                    loss_value = loss.item()
                    scale_factor = 1.0
                    if loss_value != 0:
                        power = 0
                        while abs(loss_value) > 1.0:
                            loss_value /= 10
                            power -= 1
                        while abs(loss_value) < 0.1:
                            loss_value *= 10
                            power += 1
                        scale_factor = 10 ** power
                    
                    grad = grad * scale_factor
                    
                    # Convert x0 gradient to noise gradient
                    # Since x0 = (xt - sqrt(1-αt_bar) * εt) / sqrt(αt_bar)
                    # ∂ε/∂x0 = -sqrt(αt_bar)/sqrt(1-αt_bar)
                    noise_grad = -grad * torch.sqrt(alpha_t_bar) / torch.sqrt(1 - alpha_t_bar)
                    
                    # Apply gradient to predicted noise
                    noise_pred = noise_pred - self.guidance_scale * noise_grad
                    
                    # Optional verbose output
                    if self.verbose:
                        print("="*30)
                        print(f"Timestep: {k}")
                        print(f"Loss: {loss.item():.5f}")
                        print(f"Scale factor: {scale_factor}")
                        print(f"Original noise pred: {original_noise_pred[0,:2,:2]}")
                        print(f"Noise gradient: {noise_grad[0,:2,:2]}")
                        print(f"Modified noise pred: {noise_pred[0,:2,:2]}")
                # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        return naction

    def ema_update(self):
        self.ema.step(self.net.parameters())

def wall_loss_fn(env_id, action, robot_states, safe_dist=0.07, delta_t=1.0):
    """
    differentiable wall loss function
    action: (B, T, A)
    robot_states: (B, 8)
    safe_dist: float
    delta_t: float
    """
    env_name = mu._env_names[env_id]
    wall_pos, wall_size = get_wall_position_and_size(env_name)
    wall_pos = torch.tensor(wall_pos, device=action.device, dtype=action.dtype).view(1, 1, -1)
    wall_size = torch.tensor(wall_size, device=action.device, dtype=action.dtype)/2
    wall_size = wall_size.view(1, 1, -1)
    eef_pos = [robot_states[:, -1, :3]]
    # compute trajectory of the eef
    for i in range(action.shape[1]):
        eef_pos.append(eef_pos[-1] + action[:, i, :3] * delta_t)
    eef_pos = torch.stack(eef_pos, dim=1)[:, 1:]

    # compute distance to the wall
    distances = torch.abs(eef_pos - wall_pos) - wall_size - safe_dist
    loss = torch.relu(-distances)**2
    loss = loss.sum(dim=-1).mean(dim=1)
    return loss


def get_wall_position_and_size(env_name):
    if env_name == 'button-press-topdown-v2':
        return [0.1, 0.7, 0.075], [0.1, 0.01, 0.075]
    elif env_name == 'button-press-v2':
        return [0.1, 0.6, 0.075], [0.1, 0.01, 0.075]
    elif env_name == 'pick-place-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    elif env_name == 'push-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    elif env_name == 'reach-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    else:
        raise ValueError(f"Wall doesn't exist in environment: {env_name}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
import os
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
                 n_guide_steps=4,
                 guidance_scale=0.25,
                 guide_fn=None,
                 save_dir=None,
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
        self.save_dir=save_dir
        if isinstance(guide_fn, str):  # If guide_fn is a prompt (string)
            self.guide_fn = self.generate_guide_fn(guide_fn)
        elif callable(guide_fn):  # If guide_fn is already a callable function
            self.guide_fn = guide_fn
        else:
            self.guide_fn = None
        self.verbose = verbose
        
    def extract_code_from_response(self, response):
        """
        Extracts the code block from the model's response.
        
        Args:
            response (str): The raw response string from the LLM.
        
        Returns:
            str: The extracted code block.
        """
        # Find the starting and ending markers of the code block
        start_marker = "```python"
        end_marker = "```"
        start_index = response.find(start_marker)
        end_index = response.find(end_marker, start_index + len(start_marker))
        
        if start_index == -1 or end_index == -1:
            raise ValueError("Response does not contain a valid code block.")
        
        # Extract the code block
        return response[start_index + len(start_marker):end_index].strip()

    def generate_guide_fn(self,prompt, model="gpt-4o"):
        """
        Safely generates a Python function based on the provided prompt using an LLM.

        Args:
            prompt (str): The instruction or description for the guide function.
            model (str): The LLM model to use.

        Returns:
            function: A Python function object that can be used as guide_fn.
        """
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )
        print("[INFO] Creating loss function")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in writing Python functions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        # Extract code from the response
        code = self.extract_code_from_response(response.choices[0].message.content.strip())

        save_path = os.path.join(self.save_dir, "generated_guide_fn.txt")
        with open(save_path, "w") as f:
            f.write(response.choices[0].message.content.strip())
        
        
        
        
        # Define allowed built-ins and modules
        allowed_globals = {
            "torch": torch,
            "torch.nn": torch.nn,
            "torch.Tensor": torch.Tensor,
            # Add more safe modules if necessary
        }
        
        local_vars = {}
        exec(code, allowed_globals, local_vars)
        
        if 'guide_fn' not in local_vars:
            raise ValueError("Generated code does not contain 'guide_fn'.")
        
        return local_vars['guide_fn']

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
            if self.guide_fn is not None and k >= self.noise_scheduler.timesteps[-7]:
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
                    """ if loss_value != 0:
                        power = 0
                        while abs(loss_value) > 1.0:
                            loss_value /= 10
                            power -= 1
                        while abs(loss_value) < 0.1:
                            loss_value *= 10
                            power += 1
                        scale_factor = 10 ** power """
                    #lamb=torch.sigmoid( 0.5* (k - 30))
                    grad = grad * scale_factor#*lamb
                    
                    # Convert x0 gradient to noise gradient
                    # Since x0 = (xt - sqrt(1-αt_bar) * εt) / sqrt(αt_bar)
                    # ∂ε/∂x0 = -sqrt(αt_bar)/sqrt(1-αt_bar)
                    #noise_grad = -grad * torch.sqrt(alpha_t_bar) / torch.sqrt(1 - alpha_t_bar)
                    noise_grad = -grad * torch.sqrt(1 - alpha_t_bar) / torch.sqrt(alpha_t_bar)
                    
                    # Apply gradient to predicted noise
                    noise_pred = noise_pred - self.guidance_scale * noise_grad
                    
                    # Optional verbose output
                    """ if self.verbose:
                        print("="*30)
                        print(f"Timestep: {k}")
                        print(f"Loss: {loss.item():.5f}")
                        print(f"N_guided_steps: {self.n_guide_steps}")
                        print(f"Guidance scale: {self.guidance_scale}")
                        #print(f"Scale factor: {lamb}")
                        print(f"Original noise pred: {original_noise_pred[0,:2,:2]}")
                        print(f"Noise gradient: {noise_grad[0,:2,:2]}")
                        print(f"Modified noise pred: {noise_pred[0,:2,:2]}") """
                # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        return naction

    def ema_update(self):
        self.ema.step(self.net.parameters())

def wall_loss_fn_(env_id, action, robot_states, safe_dist=0.1, delta_t=1.0):
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
    wall_size = torch.tensor(wall_size, device=action.device, dtype=action.dtype)
    wall_size = wall_size.view(1, 1, -1)
    eef_pos = [robot_states[:, -1, :3]]
    # compute trajectory of the eef
    for i in range(action.shape[1]):
        eef_pos.append(eef_pos[-1] + action[:, i, :3] * delta_t)
    eef_pos = torch.stack(eef_pos, dim=1)[:, 1:]

    """ # compute distance to the wall
    distances = torch.abs(eef_pos - wall_pos) - wall_size - safe_dist
    sharp = torch.relu(-distances)**2
    gradual = torch.exp(-distances) """
    # Compute distances for x, y, and z axes
    # Axis-specific weights
    w_x, w_y, w_z = 0.3, 0.4, 0.2  # Static weights (adjust as needed)
    distances_x = torch.abs(eef_pos[:, :, 0] - wall_pos[:, :, 0]) - wall_size[:, :, 0] 
    distances_y = torch.abs(eef_pos[:, :, 1] - wall_pos[:, :, 1]) - wall_size[:, :, 1] 
    distances_z = torch.abs(eef_pos[:, :, 2] - wall_pos[:, :, 2]) - wall_size[:, :, 2] 

    

    # Gradual and sharp penalties with axis-specific weights
    gradual_x = w_x / (1 + distances_x)**2
    gradual_y = w_y / (1 + distances_y)**2
    gradual_z = w_z / (1 + distances_z)**2

    """ sharp_x = torch.relu(-(distances_x - (safe_dist*w_x)))**2 * w_x*0.4
    sharp_y = torch.relu(-(distances_y - (safe_dist*w_y)))**2 * w_y*0.1
    sharp_z = torch.relu(-(distances_z - (safe_dist*w_z)))**2 * w_z*0.1 """

    # Combine penalties
    gradual = gradual_x + gradual_y + gradual_z
    """ sharp = sharp_x + sharp_y + sharp_z
    # Dynamic weighting: prioritize sharp penalties near the wall
    weight_sharp = torch.sigmoid(-distances.mean(dim=-1))  # More sharp weight closer to the wall
    weight_gradual = 1 - weight_sharp
    weight_sharp = weight_sharp.unsqueeze(-1)  # Shape (B, T, 1)
    weight_gradual = weight_gradual.unsqueeze(-1)  # Shape (B, T, 1) """

    horizon_weights = torch.linspace(1.0, 0.7, steps=16, device=action.device).view(1, -1, 1)
    loss = (gradual*4.25)*horizon_weights
    loss = loss.sum(dim=-1).mean(dim=1)
    return loss

def wall_loss_fn(env_id, action, robot_states, safe_dist=0.05, delta_t=1.0, cap_distance=0.3, transition_dist=0.05, cap_sharp_x=0.1, gradual_scale=1.0, pre_guidance_scale=10.0):
    """
    Differentiable wall loss function with stronger long-range guidance
    """
    env_name = mu._env_names[env_id]
    wall_pos, wall_size = get_wall_position_and_size(env_name)
    wall_pos = torch.tensor(wall_pos, device=action.device, dtype=action.dtype).view(1, 1, -1)
    wall_size = torch.tensor(wall_size, device=action.device, dtype=action.dtype).view(1, 1, -1)

    # Compute end-effector trajectory
    eef_pos = [robot_states[:, -1, :3]]
    for i in range(action.shape[1]):
        eef_pos.append(eef_pos[-1] + action[:, i, :3] * delta_t)
    eef_pos = torch.stack(eef_pos, dim=1)[:, 1:]

    # Compute distances to the wall with increased cap distance
    distances_x = torch.clip(torch.abs(eef_pos[:, :, 0] - wall_pos[:, :, 0]) - wall_size[:, :, 0], 0, cap_distance)
    distances_y = torch.clip(torch.abs(eef_pos[:, :, 1] - wall_pos[:, :, 1]) - wall_size[:, :, 1], 0, cap_distance)
    distances_z = torch.clip(torch.abs(eef_pos[:, :, 2] - wall_pos[:, :, 2]) - wall_size[:, :, 2], 0, cap_distance)

    # Unified distance calculation
    distance = torch.sqrt((distances_x / 0.2)**2 + (distances_y / 0.1)**2 + (distances_z / 0.05)**2)

    # Gradual penalty (slower decay for long-range guidance)
    gradual = torch.exp(-distance / gradual_scale) *3

    # Pre-wall penalty (far-reaching guidance term)
    pre_guidance = pre_guidance_scale / (1 + distance)**2

    # Sharp penalties (localized near the wall)
    sharp_x = torch.clip(torch.relu(-distances_x + transition_dist)**2, max=cap_sharp_x)
    sharp_y = torch.relu(-distances_y + transition_dist)**2
    sharp_z = torch.relu(-distances_z + transition_dist)**2
    sharp = sharp_x + sharp_y + sharp_z

    # Combine gradual, sharp, and pre-wall penalties
    combined_loss = gradual + sharp + pre_guidance*2

    # Apply horizon weights
    horizon_weights = torch.linspace(1.0, 0.7, steps=16, device=action.device).view(1, -1, 1)
    loss = (combined_loss * horizon_weights).sum(dim=-1).mean(dim=1)
    return loss

def speed_fn(env_id,action,robot_states):
    action=action[:,:,:3]
    speeds = torch.linalg.norm(action, dim=-1)
    max_speed = 1.0
    min_speed = 0.8

    # Penalize speeds outside the [min_speed, max_speed] range
    speed_loss = torch.maximum(speeds - max_speed, torch.tensor(0.0)) + \
                 torch.maximum(min_speed - speeds, torch.tensor(0.0))

    # Compute the mean loss per trajectory (average across the horizon): shape [batch_size]
    loss_per_trajectory = speed_loss.mean(dim=1)

    return loss_per_trajectory  # Shape: [batch_size]

def speed_fn2(env_id,action,robot_states):
    speeds = torch.linalg.norm(action, dim=-1)

    # Compute the mean speed per trajectory: shape [batch_size]
    mean_speeds = torch.mean(speeds, dim=1)

    # Define loss as negative mean speed: shape [batch_size]
    loss_per_trajectory = -mean_speeds

    return loss_per_trajectory  # Shape: [batch_size]


def get_wall_position_and_size(env_name):
    if env_name == 'button-press-topdown-v2':
        return [0.1, 0.7, 0.075], [0.1, 0.01, 0.075]
    elif env_name == 'button-press-v2':
        return [0.1, 0.6, 0.075], [0.1, 0.01, 0.075]#[0.1, 0.6, 0.075], [0.1, 0.01, 0.075]
    elif env_name == 'pick-place-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    elif env_name == 'push-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    elif env_name == 'reach-v2':
        return [0.1, 0.75, 0.06], [0.12, 0.01, 0.06]
    else:
        raise ValueError(f"Wall doesn't exist in environment: {env_name}")
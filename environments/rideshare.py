"""
NYC rideshare simulation environment wrapper.
"""

import numpy as np
from typing import Tuple, Optional, Any, Dict
from treatment_effect_gym import TreatmentEnvironment, ExperimentDesign
import jax
import jax.numpy as jnp
from functools import partial


class RideshareEnvironment(TreatmentEnvironment):
    """NYC rideshare simulation environment wrapper."""
    
    def __init__(self, 
                 n_cars: int = 300,
                 n_events: int = 10000,
                 price_control: float = 0.01,
                 price_treatment: float = 0.02,
                 w_price: float = -0.3,
                 w_eta: float = -0.005,
                 w_intercept: float = 4.0):
        """
        Args:
            n_cars: Number of cars in simulation
            n_events: Number of events to simulate
            price_control: Price per distance for control
            price_treatment: Price per distance for treatment
            w_price: Choice model price coefficient (from original paper: -0.3)
            w_eta: Choice model ETA coefficient (from original paper: -0.005)
            w_intercept: Choice model intercept (from original paper: 4.0)
        """
        self.n_cars = n_cars
        self.n_events = n_events
        self.price_control = price_control
        self.price_treatment = price_treatment
        self.w_price = w_price
        self.w_eta = w_eta
        self.w_intercept = w_intercept
        
        self._seed = None
        
        # Import JAX components (lazy import)
        self._env = None
        self._env_params = None
    
    def _init_jax_components(self):
        """Initialize JAX environment components."""
        if self._env is not None:
            return
            
        try:
            from picard.rideshare_dispatch import ManhattanRidesharePricing, SimplePricingPolicy
            import jax
            
            self._env = ManhattanRidesharePricing(n_cars=self.n_cars, n_events=self.n_events)
            self._env_params = self._env.default_params
            
            # Override with paper parameters
            self._env_params = self._env_params.replace(
                w_price=self.w_price,
                w_eta=self.w_eta,
                w_intercept=self.w_intercept
            )
            
            # Create policies
            self._policy_control = SimplePricingPolicy(
                n_cars=self.n_cars, 
                price_per_distance=self.price_control
            )
            self._policy_treatment = SimplePricingPolicy(
                n_cars=self.n_cars, 
                price_per_distance=self.price_treatment
            )
            
            # Warm up JAX compilation
            self._warm_up_jax()
            
        except ImportError as e:
            raise ImportError("JAX components not available for rideshare environment") from e
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment state."""
        if seed is not None:
            self._seed = seed
        self._init_jax_components()
    
    def _simulate_rideshare(self, policy_type: str, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate rideshare system using real JAX environment."""
        import jax
        import jax.numpy as jnp
        
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(42)
        
        self._init_jax_components()
        
        # Choose policy based on type
        if policy_type == "treatment":
            policy = self._policy_treatment
        elif policy_type == "control":
            policy = self._policy_control
        else:
            raise ValueError("Rideshare environment only supports 'treatment' or 'control' policies")
        
        # Reset environment
        obs, state = self._env.reset(key, self._env_params)
        
        rewards_list = []
        actions_list = []
        states_list = []
        
        # Run simulation for T steps (limited by available events)
        max_steps = min(T, len(self._env_params.events.t))
        
        for t in range(max_steps):
            # Extract current state information: number of available cars
            n_available = jnp.sum(state.times <= state.event.t)
            states_list.append(float(n_available))
            
            # Get action from policy
            step_key = jax.random.fold_in(key, t)
            action, action_info = policy.apply(self._env_params, dict(), obs, step_key)
            
            # Record action (pricing decision)
            actions_list.append(float(action))
            
            # Step environment
            obs, state, reward, done, info = self._env.step(step_key, state, action, self._env_params)
            rewards_list.append(float(reward))
            
            if done:
                break
        
        # Convert to numpy arrays
        rewards = np.array(rewards_list)
        actions = np.array(actions_list)
        states = np.array(states_list)
        
        # If we need more steps than available, extend with zeros
        if len(rewards) < T:
            rewards = np.pad(rewards, (0, T - len(rewards)), mode='constant')
            actions = np.pad(actions, (0, T - len(actions)), mode='constant')
            states = np.pad(states, ((0, T - len(states)), (0, 0)), mode='constant')
        
        return rewards, actions, states
    
    def simulate_treatment_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure treatment policy."""
        return self._simulate_rideshare("treatment", T, seed)
    
    def simulate_control_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure control policy."""
        return self._simulate_rideshare("control", T, seed)
    
    
    def compute_true_ate(self, T: int, seed: Optional[int] = None) -> float:
        """Compute true average treatment effect."""
        r_treat, _, _ = self.simulate_treatment_policy(T, seed)
        r_control, _, _ = self.simulate_control_policy(T, seed + 1000 if seed else None)
        return (np.sum(r_treat) - np.sum(r_control)) / T
    
    def compute_true_ate_fast(self, T: int = None, n_envs: int = 10, seed: Optional[int] = None, n_events: Optional[int] = None) -> float:
        """Compute true ATE using JAX vectorization for speed.
        
        Args:
            T: Number of time steps (for backward compatibility)
            n_envs: Number of parallel environments for averaging
            seed: Random seed
            n_events: Number of ride request events to use. If provided, overrides T.
        """
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(42)
            
        self._init_jax_components()
        
        # Determine simulation length: use event-based if provided, otherwise step-based
        if n_events is not None:
            # Event-based ATE computation: use same number of events as main simulation
            available_events = len(self._env_params.events.t)
            simulation_length = min(n_events, available_events)
            print(f"Computing true ATE with {simulation_length} ride request events")
        else:
            # Step-based ATE computation (backward compatibility)
            if T is None:
                raise ValueError("Either T (steps) or n_events (ride requests) must be provided")
            simulation_length = T
            print(f"Computing true ATE with {simulation_length} steps")
        
        # Run vectorized simulations
        treat_rewards = self._run_vectorized_simulation("treatment", simulation_length, n_envs, key)
        control_key = jax.random.fold_in(key, 10000)  # Different key for control
        control_rewards = self._run_vectorized_simulation("control", simulation_length, n_envs, control_key)
        
        # Compute ATE: average treatment effect per ride request event
        # treat_rewards and control_rewards have shape (n_envs, simulation_length)
        # Each element [i,j] is the revenue from the j-th ride request in the i-th environment
        
        treat_total = jnp.mean(jnp.sum(treat_rewards, axis=1))  # Mean total revenue per environment
        control_total = jnp.mean(jnp.sum(control_rewards, axis=1))  # Mean total revenue per environment
        
        # ATE = (total treatment revenue - total control revenue) / number of ride requests
        # This gives the average revenue difference per ride request
        return float((treat_total - control_total) / simulation_length)
    
    def _run_vectorized_simulation(self, policy_type: str, T: int, n_envs: int, key: jax.Array) -> jnp.ndarray:
        """Run vectorized simulation using JAX for speed."""
        # Create stepper function from original paper approach
        stepper_fn = self._create_jax_stepper(policy_type)
        
        # Initialize environments
        reset_keys = jax.random.split(key, n_envs)
        initial_states = jax.vmap(self._env.reset, in_axes=(0, None))(reset_keys, self._env_params)
        
        # Create step keys for each environment and time step
        step_keys = jax.random.split(jax.random.fold_in(key, 1), n_envs * T).reshape(n_envs, T, 2)
        
        # Run vectorized simulation
        max_steps = min(T, len(self._env_params.events.t))
        
        def scan_fn(carry, step_info):
            obs, state = carry
            step_key, t = step_info
            
            # Get action from policy
            if policy_type == "treatment":
                policy = self._policy_treatment
            else:
                policy = self._policy_control
                
            action, _ = policy.apply(self._env_params, dict(), obs, step_key)
            
            # Step environment
            new_obs, new_state, reward, done, _ = self._env.step(step_key, state, action, self._env_params)
            
            return (new_obs, new_state), reward
        
        # Vectorize across environments
        vmapped_scan = jax.vmap(
            lambda init_state, keys: jax.lax.scan(
                scan_fn,
                init_state,
                (keys, jnp.arange(max_steps))
            )[1],  # Only return rewards
            in_axes=(0, 0)
        )
        
        rewards = vmapped_scan(initial_states, step_keys[:, :max_steps])
        
        # Pad if necessary
        if max_steps < T:
            padding = jnp.zeros((n_envs, T - max_steps))
            rewards = jnp.concatenate([rewards, padding], axis=1)
        
        return rewards
    
    def _create_jax_stepper(self, policy_type: str):
        """Create JAX-compiled stepper function."""
        if policy_type == "treatment":
            policy = self._policy_treatment
        else:
            policy = self._policy_control
            
        @partial(jax.jit, static_argnums=(0,))
        def stepper(env, env_params, carry, step_info):
            obs, state = carry
            step_key, t = step_info
            
            action, _ = policy.apply(env_params, dict(), obs, step_key)
            new_obs, new_state, reward, done, _ = env.step(step_key, state, action, env_params)
            
            return (new_obs, new_state), reward
            
        return partial(stepper, self._env, self._env_params)
    
    def _warm_up_jax(self):
        """Pre-compile JAX functions to eliminate compilation overhead."""
        try:
            # Warm up with small simulation
            key = jax.random.PRNGKey(0)
            small_rewards = self._run_vectorized_simulation("treatment", 5, 2, key)
            # This forces JAX to compile the functions
        except Exception:
            # If warm-up fails, continue without it
            pass
    
    def simulate_experiment(self, design: ExperimentDesign, T: int = None, seed: Optional[int] = None, n_envs: int = 1, chunk_size: int = 1000, n_events: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """JAX-optimized experimental simulation with vectorization and chunked processing.
        
        Args:
            design: Experimental design for treatment assignment
            T: Number of time steps (for backward compatibility)
            seed: Random seed
            n_envs: Number of parallel environments (default=1 for benchmark compatibility)
            chunk_size: Chunk size for memory-efficient processing
            n_events: Number of ride request events to simulate. If provided, overrides T.
                     Examples: 
                     - 1000 = 1000 ride requests
                     - 10000 = 10000 ride requests (demo setting)
                     - Uses first n_events from historical data
        
        Returns:
            Tuple of (rewards, actions, states, simulation_times) where simulation_times
            are the actual timestamps used for treatment assignment (for proper aggregation)
        """
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(42)
            
        self._init_jax_components()
        design.reset(seed)
        
        # Determine simulation length: use event-based if provided, otherwise step-based
        if n_events is not None:
            # Event-based trajectory control: simulate first n_events from historical data
            available_events = len(self._env_params.events.t)
            max_steps = min(n_events, available_events)
            
            event_times = self._env_params.events.t[:max_steps]
            duration = (event_times[-1] - event_times[0]) / 3600  # Convert to hours
            print(f"Event-based simulation: {max_steps} ride requests ({duration:.1f} hours of historical data)")
        else:
            # Step-based trajectory control (backward compatibility)
            if T is None:
                raise ValueError("Either T (steps) or n_events (ride requests) must be provided")
            max_steps = min(T, len(self._env_params.events.t))
            print(f"Step-based simulation: {max_steps} steps")
        
        # Get actual simulation times used for treatment assignment
        actual_simulation_times = np.array(self._env_params.events.t[:max_steps])
        
        # Optimized treatment assignment generation using JAX
        treatment_matrix = self._generate_treatment_assignments_vectorized(design, max_steps, n_envs, key)
        
        # Run chunked vectorized simulation with treatment assignments
        rewards, actions, states = self._run_chunked_vectorized_experiment(treatment_matrix, max_steps, n_envs, key, chunk_size)
        
        # Convert to numpy and return (flatten across environments for benchmark compatibility)
        rewards_flat = np.array(rewards).flatten()[:max_steps]  
        actions_flat = np.array(actions).flatten()[:max_steps]
        states_flat = np.array(states).flatten()[:max_steps]  # States are scalars, so just flatten
        
        return rewards_flat, actions_flat, states_flat
    
    def get_simulation_times(self, n_events: Optional[int] = None, T: Optional[int] = None) -> np.ndarray:
        """Get the actual simulation times used for treatment assignment.
        
        This is needed for proper switchback interval aggregation since treatment
        assignments are based on actual historical taxi event timestamps, not step indices.
        """
        self._init_jax_components()
        
        if n_events is not None:
            available_events = len(self._env_params.events.t)
            max_steps = min(n_events, available_events)
        else:
            max_steps = min(T, len(self._env_params.events.t))
        
        return np.array(self._env_params.events.t[:max_steps])
    
    def _generate_treatment_assignments_vectorized(self, design: ExperimentDesign, max_steps: int, n_envs: int, key: jax.Array) -> jnp.ndarray:
        """Generate treatment assignments efficiently using JAX operations."""
        from experiment_designs.randomized import RandomizedDesign
        from experiment_designs.switchback import SwitchbackDesign
        
        if isinstance(design, RandomizedDesign):
            # Generate all random assignments at once
            treat_key = jax.random.fold_in(key, 12345)
            assignments = jax.random.bernoulli(treat_key, design.p, (n_envs, max_steps))
            return assignments.astype(jnp.int32)
            
        elif isinstance(design, SwitchbackDesign):
            # Use actual simulation times from rideshare events (like original paper)
            event_times = self._env_params.events.t[:max_steps]  # Actual simulation timestamps
            
            # Calculate period IDs based on actual simulation time
            period_ids = event_times // design.switch_every
            unique_periods, period_indices = jnp.unique(period_ids, return_inverse=True)
            
            # Generate random assignments for each unique period
            switch_key = jax.random.fold_in(key, 54321)
            period_assignments = jax.random.bernoulli(switch_key, design.p, (n_envs, len(unique_periods)))
            
            # Map period assignments back to time steps using period indices
            assignments = period_assignments[:, period_indices]
            return assignments.astype(jnp.int32)
        else:
            # Fallback to original method for other designs
            treatment_assignments = []
            for env_idx in range(n_envs):
                env_assignments = []
                for t in range(max_steps):
                    assignment = design.assign_treatment(t, t, max_time=max_steps)
                    env_assignments.append(assignment)
                treatment_assignments.append(env_assignments)
            return jnp.array(treatment_assignments)
    
    def _run_chunked_vectorized_experiment(self, treatment_matrix: jnp.ndarray, T: int, n_envs: int, key: jax.Array, chunk_size: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run vectorized experiment with chunked processing for memory efficiency."""
        max_steps = min(T, len(self._env_params.events.t))
        
        # Initialize environments
        reset_keys = jax.random.split(key, n_envs)
        initial_states = jax.vmap(self._env.reset, in_axes=(0, None))(reset_keys, self._env_params)
        
        # Create step keys
        step_keys = jax.random.split(jax.random.fold_in(key, 1), n_envs * max_steps).reshape(n_envs, max_steps, 2)
        
        # Calculate number of chunks
        n_chunks = (max_steps + chunk_size - 1) // chunk_size
        
        all_rewards_chunks = []
        all_actions_chunks = []
        all_states_chunks = []
        
        current_carry = initial_states
        
        print(f"Processing {n_chunks} chunks for memory efficiency...")
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_steps)
            current_chunk_size = chunk_end - chunk_start
            
            # Extract chunk data
            chunk_keys = step_keys[:, chunk_start:chunk_end]
            chunk_treatments = treatment_matrix[:, chunk_start:chunk_end]
            chunk_times = jnp.arange(chunk_start, chunk_end)
            
            # Process chunk using optimized stepper
            chunk_carry, chunk_outputs = self._process_experiment_chunk(
                current_carry, chunk_keys, chunk_treatments, chunk_times
            )
            
            # Force computation and memory cleanup
            chunk_carry[0].block_until_ready()  # obs part
            
            chunk_rewards, chunk_actions, chunk_states = chunk_outputs
            all_rewards_chunks.append(chunk_rewards)
            all_actions_chunks.append(chunk_actions)
            all_states_chunks.append(chunk_states)
            
            current_carry = chunk_carry
            
            if chunk_idx % 5 == 0:
                print(f"  Completed chunk {chunk_idx + 1}/{n_chunks}")
        
        # Concatenate all chunks
        final_rewards = jnp.concatenate(all_rewards_chunks, axis=1)
        final_actions = jnp.concatenate(all_actions_chunks, axis=1)
        final_states = jnp.concatenate(all_states_chunks, axis=1)
        
        # Pad if necessary
        if max_steps < T:
            pad_size = T - max_steps
            final_rewards = jnp.concatenate([final_rewards, jnp.zeros((n_envs, pad_size))], axis=1)
            final_actions = jnp.concatenate([final_actions, jnp.zeros((n_envs, pad_size))], axis=1)
            final_states = jnp.concatenate([final_states, jnp.zeros((n_envs, pad_size, final_states.shape[-1]))], axis=1)
        
        return final_rewards, final_actions, final_states
    
    @partial(jax.jit, static_argnums=(0,))
    def _process_experiment_chunk(self, carry, chunk_keys, chunk_treatments, chunk_times):
        """Process a single chunk of the experiment using JAX scan."""
        def scan_fn(carry_state, step_inputs):
            obs, state = carry_state
            step_key, is_treatment, t = step_inputs
            
            # Conditional policy selection using jax.lax.cond for efficiency
            action_treat, _ = self._policy_treatment.apply(self._env_params, dict(), obs, step_key)
            action_control, _ = self._policy_control.apply(self._env_params, dict(), obs, step_key)
            action = jax.lax.select(is_treatment, action_treat, action_control)
            
            # Step environment
            new_obs, new_state, reward, done, _ = self._env.step(step_key, state, action, self._env_params)
            
            # Extract state information: number of available cars
            n_available = jnp.sum(state.times <= state.event.t)
            state_info = n_available  # Number of available cars (scalar state)
            
            return (new_obs, new_state), (reward, is_treatment, state_info)
        
        # Vectorize scan across all environments
        vmapped_scan = jax.vmap(
            lambda carry_env, keys_env, treats_env, times_env: jax.lax.scan(
                scan_fn,
                carry_env,
                (keys_env, treats_env, times_env)
            ),
            in_axes=(0, 0, 0, None)
        )
        
        return vmapped_scan(carry, chunk_keys, chunk_treatments, chunk_times)
    
    def simulate_batch_experiments(self, design: ExperimentDesign, T: int = None, n_trials: int = 100, seed: Optional[int] = None, batch_size: int = 100, chunk_size: int = 1000, n_events: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast batch simulation using original paper's vectorized approach."""
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(42)
            
        self._init_jax_components()
        
        print(f"Running {n_trials} trials in batches of {batch_size} with chunk size {chunk_size}")
        
        all_rewards = []
        all_actions = []
        all_states = []
        
        # Process trials in batches for memory management
        n_batches = (n_trials + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_trials)
            current_batch_size = end_idx - start_idx
            
            print(f"Processing batch {batch_idx + 1}/{n_batches} ({current_batch_size} trials)...")
            
            # Generate batch-specific key
            batch_key = jax.random.fold_in(key, batch_idx)
            
            # Run vectorized batch experiment (inspired by original run_trials function)
            batch_rewards, batch_actions, batch_states = self._run_vectorized_batch_trials(
                design, T, current_batch_size, batch_key, chunk_size, n_events
            )
            
            # Convert to list format for compatibility
            for trial_idx in range(current_batch_size):
                all_rewards.append(np.array(batch_rewards[trial_idx]))
                all_actions.append(np.array(batch_actions[trial_idx]))
                all_states.append(np.array(batch_states[trial_idx]))
        
        return all_rewards, all_actions, all_states
    
    def _run_vectorized_batch_trials(self, design: ExperimentDesign, T: int, n_envs: int, key: jax.Array, chunk_size: int, n_events: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run vectorized batch trials using techniques from original paper."""
        # Determine simulation length: use event-based if provided, otherwise step-based
        if n_events is not None:
            # Event-based trajectory control
            available_events = len(self._env_params.events.t)
            max_steps = min(n_events, available_events)
        else:
            # Step-based trajectory control
            max_steps = min(T, len(self._env_params.events.t))
        
        # Generate treatment assignments for all environments (vectorized)
        treatment_matrix = self._generate_treatment_assignments_vectorized(design, max_steps, n_envs, key)
        
        # Initialize environments in parallel
        reset_key, step_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, n_envs)
        step_keys = jax.random.split(step_key, n_envs * max_steps).reshape(n_envs, max_steps, 2)
        
        # Use chunked processing approach from original paper
        obs_and_states_initial = jax.vmap(self._env.reset, in_axes=(0, None))(reset_keys, self._env_params)
        
        # Create experiment info structure similar to original
        from jaxtyping import Integer, Bool
        from flax import struct
        
        @struct.dataclass
        class ExperimentInfo:
            t: Integer[jax.Array, "n_steps"] 
            is_treat: Bool[jax.Array, "n_steps"]
            key: jax.Array
        
        infos = ExperimentInfo(
            t=jnp.tile(jnp.arange(max_steps).reshape(1, -1), (n_envs, 1)),
            is_treat=treatment_matrix.astype(bool),
            key=step_keys,
        )
        
        # Process using chunked scanner approach
        rewards, actions, states = self._chunked_scanner(obs_and_states_initial, infos, chunk_size)
        
        # Pad if necessary (only for step-based control)
        if T is not None and max_steps < T:
            pad_size = T - max_steps
            rewards = jnp.concatenate([rewards, jnp.zeros((n_envs, pad_size))], axis=1)
            actions = jnp.concatenate([actions, jnp.zeros((n_envs, pad_size))], axis=1)
            states = jnp.concatenate([states, jnp.zeros((n_envs, pad_size, states.shape[-1]))], axis=1)
        
        return rewards, actions, states
    
    def _chunked_scanner(self, obs_and_states_initial, infos, chunk_size):
        """Chunked scanner implementation based on original paper approach."""
        def scan_fn_for_one_batch_element_one_chunk(carry_obs_state, infos_chunk):
            """Process one chunk for one environment."""
            def stepper_fn(carry, info_step):
                obs, state = carry
                step_key = info_step.key
                is_treat = info_step.is_treat
                
                # Conditional policy selection (original paper approach)
                action_treat, _ = self._policy_treatment.apply(self._env_params, dict(), obs, step_key)
                action_control, _ = self._policy_control.apply(self._env_params, dict(), obs, step_key)
                action = jax.lax.select(is_treat, action_treat, action_control)
                
                # Step environment
                new_obs, new_state, reward, _, _ = self._env.step(step_key, state, action, self._env_params)
                
                # Extract state info: number of available cars
                n_available = jnp.sum(new_state.times <= new_state.event.t)
                state_info = n_available
                
                return (new_obs, new_state), (reward, is_treat.astype(jnp.int32), state_info)
            
            final_carry, outputs = jax.lax.scan(stepper_fn, carry_obs_state, infos_chunk)
            return final_carry, outputs
        
        # Vectorize across all environments
        vmapped_chunk_processor = jax.vmap(
            scan_fn_for_one_batch_element_one_chunk,
            in_axes=(0, 0),
            out_axes=0
        )
        
        current_carry = obs_and_states_initial
        n_steps_total = infos.t.shape[1]
        n_chunks = (n_steps_total + chunk_size - 1) // chunk_size
        
        all_rewards_chunks = []
        all_actions_chunks = []
        all_states_chunks = []
        
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_steps_total)
            
            # Extract chunk data
            from jaxtyping import Integer, Bool
            from flax import struct
            
            @struct.dataclass
            class ExperimentInfo:
                t: Integer[jax.Array, "n_steps"]
                is_treat: Bool[jax.Array, "n_steps"] 
                key: jax.Array
            
            current_chunk_infos = ExperimentInfo(
                t=infos.t[:, chunk_start:chunk_end],
                is_treat=infos.is_treat[:, chunk_start:chunk_end],
                key=infos.key[:, chunk_start:chunk_end],
            )
            
            # Process chunk
            current_carry, chunk_outputs = vmapped_chunk_processor(current_carry, current_chunk_infos)
            current_carry[0].block_until_ready()  # Force computation
            
            # Collect outputs
            rewards_chunk, actions_chunk, states_chunk = chunk_outputs
            all_rewards_chunks.append(rewards_chunk)
            all_actions_chunks.append(actions_chunk)
            all_states_chunks.append(states_chunk)
        
        # Concatenate all chunks
        final_rewards = jnp.concatenate(all_rewards_chunks, axis=1)
        final_actions = jnp.concatenate(all_actions_chunks, axis=1)
        final_states = jnp.concatenate(all_states_chunks, axis=1)
        
        return final_rewards, final_actions, final_states
    
    @property
    def name(self) -> str:
        return "Rideshare"
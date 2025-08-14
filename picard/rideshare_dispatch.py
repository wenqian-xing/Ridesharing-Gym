"""
Pricing and dispatch ridesharing environments
"""
from dataclasses import field
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
from flax import linen as nn
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from jaxtyping import Float, Integer, Bool
import funcy as f
# import pooch  # No longer needed - using local data files
import numpy as np
import pandas as pd

from picard.nn import Policy, MLP


# Driver state constants for enhanced driver dynamics
DRIVER_STATE_OFFLINE = 0
DRIVER_STATE_ONLINE_IDLE = 1
DRIVER_STATE_EN_ROUTE_PICKUP = 2
DRIVER_STATE_ON_TRIP = 3
DRIVER_STATE_REPOSITIONING = 4
DRIVER_STATE_ON_BREAK = 5

# Experience level constants
EXPERIENCE_NEW = 0
EXPERIENCE_EXPERIENCED = 1  
EXPERIENCE_VETERAN = 2

# Shift pattern constants
SHIFT_FULL_TIME = 0
SHIFT_PART_TIME_DAY = 1
SHIFT_PART_TIME_EVENING = 2
SHIFT_WEEKEND = 3


def should_drivers_go_online(current_time, state: "EnvState", params: "PricingEnvParams", key: chex.PRNGKey) -> Integer[Array, "n_cars"]:
    """Determine which offline drivers should go online based on shift patterns and market conditions."""
    hour_of_day = jnp.floor(current_time / 3600) % 24
    
    # Access driver parameters from dispatch_env_params
    dispatch_params = params.dispatch_env_params
    
    # Check if drivers are in their preferred shift time
    in_shift_time = jnp.select([
        dispatch_params.driver_shift_patterns == SHIFT_FULL_TIME,
        dispatch_params.driver_shift_patterns == SHIFT_PART_TIME_DAY,
        dispatch_params.driver_shift_patterns == SHIFT_PART_TIME_EVENING,
        dispatch_params.driver_shift_patterns == SHIFT_WEEKEND,
    ], [
        (hour_of_day >= 6) & (hour_of_day <= 22),  # 6 AM - 10 PM
        (hour_of_day >= 6) & (hour_of_day <= 18),  # 6 AM - 6 PM
        (hour_of_day >= 17) | (hour_of_day <= 2),  # 5 PM - 2 AM
        (hour_of_day >= 10) & (hour_of_day <= 24), # 10 AM - 12 AM
    ])
    
    # Check daily limits
    under_hours_limit = state.total_session_times < (dispatch_params.daily_hours_targets * 3600)
    under_earnings_limit = state.daily_earnings < dispatch_params.daily_earnings_targets
    
    # Only consider offline drivers
    is_offline = state.driver_states == DRIVER_STATE_OFFLINE
    
    # Base probability of going online
    base_prob = 0.7
    can_go_online = is_offline & in_shift_time & under_hours_limit & under_earnings_limit
    
    # Random decisions
    n_drivers = len(state.driver_states)
    keys = jax.random.split(key, n_drivers)
    random_values = jax.vmap(jax.random.uniform)(keys)
    should_go_online = can_go_online & (random_values < base_prob)
    
    return should_go_online.astype(jnp.int32)


def should_drivers_accept_trip(trip_eta, state: "EnvState", params: "PricingEnvParams", key: chex.PRNGKey) -> Integer[Array, "n_cars"]:
    """Determine which online idle drivers should accept a trip request."""
    
    # Access driver parameters from dispatch_env_params
    dispatch_params = params.dispatch_env_params
    
    # Base acceptance rates
    acceptance_prob = dispatch_params.acceptance_rate_baselines
    
    # ETA penalty - longer ETAs reduce acceptance
    eta_penalty = jnp.where(
        trip_eta > 600,  # > 10 min
        0.7,
        jnp.where(trip_eta > 300, 0.85, 1.0)  # > 5 min
    )
    acceptance_prob = acceptance_prob * eta_penalty
    
    # Experience factor
    exp_factor = jnp.select([
        dispatch_params.driver_experience_levels == EXPERIENCE_NEW,
        dispatch_params.driver_experience_levels == EXPERIENCE_EXPERIENCED,
        dispatch_params.driver_experience_levels == EXPERIENCE_VETERAN,
    ], [1.2, 1.0, 0.9])  # New drivers accept more, veterans are selective
    acceptance_prob = acceptance_prob * exp_factor
    
    # Consecutive rejection penalty (platform pressure)
    rejection_penalty = jnp.where(
        state.consecutive_rejections > 3,
        1.5,  # Increased pressure to accept
        1.0
    )
    acceptance_prob = acceptance_prob * rejection_penalty
    
    # Multi-platform competition
    platform_factor = jnp.where(
        dispatch_params.uses_multiple_platforms,
        0.8,  # Slightly lower acceptance if using multiple platforms
        1.0
    )
    acceptance_prob = acceptance_prob * platform_factor
    
    # Only online idle drivers can accept
    can_accept = state.driver_states == DRIVER_STATE_ONLINE_IDLE
    final_prob = jnp.where(can_accept, jnp.clip(acceptance_prob, 0.0, 1.0), 0.0)
    
    # Random decisions
    n_drivers = len(state.driver_states)
    keys = jax.random.split(key, n_drivers)
    random_values = jax.vmap(jax.random.uniform)(keys)
    should_accept = (random_values < final_prob) & can_accept
    
    return should_accept.astype(jnp.int32)


def update_driver_states(current_time, old_state: "EnvState", trip_accepted, 
                        selected_driver, trip_reward) -> "EnvState":
    """Update driver states after a simulation step."""
    
    # Update session times for online drivers
    online_mask = old_state.driver_states != DRIVER_STATE_OFFLINE
    time_delta = current_time - old_state.session_start_times
    new_session_times = jnp.where(
        online_mask,
        old_state.total_session_times + time_delta,
        old_state.total_session_times
    )
    
    # Update time since last trip
    new_time_since_trip = jnp.where(
        online_mask,
        old_state.time_since_last_trip + time_delta,
        old_state.time_since_last_trip
    )
    
    # Update driver states and metrics using JAX control flow
    new_driver_states = old_state.driver_states
    new_daily_earnings = old_state.daily_earnings
    new_trips_completed = old_state.trips_completed
    new_consecutive_rejections = old_state.consecutive_rejections
    
    # Use JAX conditionals instead of Python if statements
    trip_accepted_and_valid = trip_accepted & (selected_driver >= 0)
    
    # Create selection mask for the driver
    driver_mask = jnp.arange(len(new_driver_states)) == selected_driver
    
    # Update selected driver who accepted the trip using vectorized operations
    new_driver_states = jnp.where(
        driver_mask & trip_accepted_and_valid,
        DRIVER_STATE_ON_TRIP,
        new_driver_states
    )
    new_daily_earnings = jnp.where(
        driver_mask & trip_accepted_and_valid,
        new_daily_earnings + trip_reward,
        new_daily_earnings
    )
    new_trips_completed = jnp.where(
        driver_mask & trip_accepted_and_valid,
        new_trips_completed + 1,
        new_trips_completed
    )
    new_consecutive_rejections = jnp.where(
        driver_mask & trip_accepted_and_valid,
        0,
        new_consecutive_rejections
    )
    new_time_since_trip = jnp.where(
        driver_mask & trip_accepted_and_valid,
        0.0,
        new_time_since_trip
    )
    
    return old_state.replace(
        driver_states=new_driver_states,
        total_session_times=new_session_times,
        daily_earnings=new_daily_earnings,
        trips_completed=new_trips_completed,
        consecutive_rejections=new_consecutive_rejections,
        time_since_last_trip=new_time_since_trip
    )


@struct.dataclass
class RideshareEvent:
    t: Integer[Array, "n_events"]
    src: Integer[Array, "n_events"]
    dest: Integer[Array, "n_events"]


@struct.dataclass
class EnvState(environment.EnvState):
    locations: Integer[
        Array, "n_cars"
    ]  # Ending point of the car's most recent trip
    times: Integer[Array, "n_cars"]  # Ending time of the car's most recent trip
    key: Integer[Array, "2"]
    event: RideshareEvent
    
    # Enhanced driver dynamics state
    driver_states: Integer[Array, "n_cars"]  # DriverState enum values (0=offline, 1=online_idle, etc.)
    session_start_times: Float[Array, "n_cars"]  # When driver went online today
    total_session_times: Float[Array, "n_cars"]  # Total time online today
    daily_earnings: Float[Array, "n_cars"]  # Earnings accumulated today
    trips_completed: Integer[Array, "n_cars"]  # Number of trips completed today
    consecutive_rejections: Integer[Array, "n_cars"]  # Consecutive trip rejections
    time_since_last_trip: Float[Array, "n_cars"]  # Time since last completed trip
    assigned_requests: Integer[Array, "n_cars"]  # Request ID (-1 if none)


@struct.dataclass
class EnvParams(environment.EnvParams):
    events: RideshareEvent = RideshareEvent(
        jnp.zeros(1), jnp.zeros(1), jnp.zeros(1)
    )
    distances: Integer[Array, "nodes nodes"] = field(
        default_factory=lambda: jnp.zeros((1, 1))
    )
    n_cars: int = 1
    
    # Enhanced driver profile parameters
    driver_experience_levels: Integer[Array, "n_cars"] = field(
        default_factory=lambda: jnp.zeros(1)
    )  # 0=new, 1=experienced, 2=veteran
    driver_shift_patterns: Integer[Array, "n_cars"] = field(
        default_factory=lambda: jnp.zeros(1)
    )  # 0=full_time, 1=part_time_day, 2=part_time_evening, 3=weekend
    driver_home_locations: Integer[Array, "n_cars"] = field(
        default_factory=lambda: jnp.zeros(1)
    )  # Home base location for each driver
    daily_hours_targets: Float[Array, "n_cars"] = field(
        default_factory=lambda: jnp.ones(1) * 6.0
    )  # Target hours per day
    daily_earnings_targets: Float[Array, "n_cars"] = field(
        default_factory=lambda: jnp.ones(1) * 120.0
    )  # Target earnings per day
    acceptance_rate_baselines: Float[Array, "n_cars"] = field(
        default_factory=lambda: jnp.ones(1) * 0.75
    )  # Base acceptance rates
    uses_multiple_platforms: Bool[Array, "n_cars"] = field(
        default_factory=lambda: jnp.zeros(1, dtype=bool)
    )  # Whether driver uses multiple platforms

    @property
    def n_nodes(self) -> int:
        return self.distances.shape[0]

    @property
    def n_events(self) -> int:
        return self.events.t.shape[0]


@struct.dataclass
class PricingEnvParams(environment.EnvParams):
    dispatch_env_params: EnvParams = field(default_factory=lambda: EnvParams())
    w_price: float = -1.0
    w_eta: float = -1.0
    w_intercept: float = 1.0

    @property
    def events(self) -> RideshareEvent:
        return self.dispatch_env_params.events

    @property
    def distances(self) -> Integer[Array, "nodes nodes"]:
        return self.dispatch_env_params.distances

    @property
    def n_cars(self) -> int:
        return self.dispatch_env_params.n_cars


def get_nth_event(event: RideshareEvent, n: int) -> RideshareEvent:
    return RideshareEvent(event.t[n], event.src[n], event.dest[n])


@partial(jax.jit, static_argnums=(0,))
def obs_to_state(n_cars: int, obs: Integer[Array, "o_dim"]):
    obs = obs.astype(int)
    locations = obs[3 : 3 + n_cars]
    times = obs[3 + n_cars :]
    event = RideshareEvent(obs[0], obs[1], obs[2])
    return event, locations, times


class RideshareDispatch(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self, n_cars: int = 100, n_nodes: int = 100, n_events: int = 1000
    ):
        super(RideshareDispatch, self).__init__()
        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.n_events = n_events

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        return jax.lax.cond(
            action >= 0,
            lambda: self.step_env_dispatch(key, state, action, params),
            lambda: self.step_env_unfulfill(key, state, action, params),
        )

    def step_env_unfulfill(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        next_event = get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            locations=state.locations,
            times=state.times,
            key=state.key,
            event=next_event,
        )
        done = self.is_terminal(next_state, params)
        reward = 0.0  # Should maybe have some unfulfill cost
        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    def step_env_dispatch(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        t_pickup, _ = self.get_pickup_and_dropoff_times(state, action, params)
        next_state = self.dispatch_and_update_state(state, action, params)
        done = self.is_terminal(next_state, params)
        reward = -(t_pickup - state.event.t)  # Minimize rider wait times

        return (
            lax.stop_gradient(self.get_obs(next_state)),
            lax.stop_gradient(next_state),
            jnp.array(reward, dtype=float),
            done,
            {"discount": self.discount(state, params)},
        )

    @staticmethod
    def get_pickup_and_dropoff_times(
        state: EnvState, car_id: int, params: EnvParams
    ) -> Tuple[int, int]:
        """
        Compute the pickup and dropoff times for dispatching a `car_id`
        to serve `state.event`.
        """
        event = state.event
        t_pickup = (
            jnp.maximum(event.t, state.times[car_id])
            + params.distances[state.locations[car_id], event.src]
        )
        t_dropoff = t_pickup + params.distances[event.src, event.dest]
        return t_pickup, t_dropoff

    def dispatch_and_update_state(
        self, state: EnvState, car_id: int, params: EnvParams
    ) -> EnvState:
        event = state.event
        t_pickup, t_dropoff = self.get_pickup_and_dropoff_times(
            state, car_id, params
        )
        # is_accept = params.choice_model(choice_key, event, price, etd - event.t)
        new_locations = state.locations.at[car_id].set(event.dest)
        new_times = state.times.at[car_id].set(t_dropoff)
        next_event = get_nth_event(params.events, state.time + 1)
        next_state = EnvState(
            time=state.time + 1,
            locations=new_locations,
            times=new_times,
            key=state.key,
            event=next_event,
        )
        return next_state

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_reset, key_online = jax.random.split(key, 3)
        
        # Initialize basic state
        initial_locations = jax.random.choice(
            key_reset, jnp.arange(self.n_nodes), (self.n_cars,)
        )
        
        # Enhanced driver dynamics: start some drivers online (realistic initialization)
        online_probability = 0.2  # 20% of drivers start online
        initial_driver_states = jnp.where(
            jax.random.bernoulli(key_online, online_probability, (self.n_cars,)),
            DRIVER_STATE_ONLINE_IDLE,
            DRIVER_STATE_OFFLINE
        )
        
        state = EnvState(
            time=0,
            locations=initial_locations,
            times=jnp.zeros(self.n_cars, dtype=int),  # Empty cars
            key=key,
            event=get_nth_event(params.events, 0),
            
            # Enhanced driver dynamics initialization
            driver_states=initial_driver_states,
            session_start_times=jnp.zeros(self.n_cars),
            total_session_times=jnp.zeros(self.n_cars),
            daily_earnings=jnp.zeros(self.n_cars),
            trips_completed=jnp.zeros(self.n_cars, dtype=int),
            consecutive_rejections=jnp.zeros(self.n_cars, dtype=int),
            time_since_last_trip=jnp.zeros(self.n_cars),
            assigned_requests=jnp.full(self.n_cars, -1, dtype=int),
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Applies observation function to state."""
        return jnp.concatenate(
            [
                jnp.reshape(state.event.t, (1,)),
                jnp.reshape(state.event.src, (1,)),
                jnp.reshape(state.event.dest, (1,)),
                state.locations,
                state.times,
            ]
        )

    def is_terminal(self, state: EnvState, params=None) -> jnp.ndarray:
        """Check whether state is terminal."""
        return state.time >= self.n_events

    @property
    def name(self) -> str:
        """Environment name."""
        return "RideshareDispatch-v0"

    @property
    def num_actions(self) -> int:
        return self.n_cars

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(params.n_cars)

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Observation space of the environment."""
        return spaces.Box(0, jnp.inf, (3 + 2 * self.n_cars), jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "locations": spaces.Box(
                    0, params.n_nodes, (self.n_cars,), jnp.int32
                ),
                "times": spaces.Box(
                    0, params.events.t[-1], (self.n_cars,), jnp.int32
                ),
                "event": spaces.Dict(
                    {
                        "t": spaces.Box(
                            0, params.events.t[-1], (1,), jnp.int32
                        ),
                        "src": spaces.Box(0, self.n_nodes, (1,), jnp.int32),
                        "dest": spaces.Box(0, self.n_nodes, (1,), jnp.int32),
                    }
                ),
            }
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            events=RideshareEvent(
                jnp.arange(self.n_events),
                jax.random.choice(
                    jax.random.PRNGKey(0),
                    jnp.arange(self.n_nodes),
                    (self.n_events,),
                ),
                jax.random.choice(
                    jax.random.PRNGKey(1),
                    jnp.arange(self.n_nodes),
                    (self.n_events,),
                ),
            ),
            distances=jax.random.normal(
                jax.random.PRNGKey(2), (self.n_nodes, self.n_nodes)
            ),
            n_cars=self.n_cars,
        )


class RidesharePricing(RideshareDispatch):
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ):
        # Enhanced driver dynamics: first check who should go online
        go_online_key, driver_decision_key, step_key = jax.random.split(key, 3)
        
        # Update drivers who should go online
        should_go_online = should_drivers_go_online(
            state.event.t, state, params, go_online_key
        )
        updated_driver_states = jnp.where(
            should_go_online,
            DRIVER_STATE_ONLINE_IDLE,
            state.driver_states
        )
        
        # Update state with new online drivers
        current_state = state.replace(driver_states=updated_driver_states)
        
        # Calculate ETAs only for online idle drivers
        online_idle_mask = current_state.driver_states == DRIVER_STATE_ONLINE_IDLE
        
        # Compute ETAs for all drivers, but only consider online idle ones
        etas = (
            jnp.maximum(current_state.event.t, current_state.times)
            + params.distances[current_state.locations, current_state.event.src]
            - current_state.event.t
        )
        
        # Set ETA to infinity for offline/unavailable drivers
        available_etas = jnp.where(online_idle_mask, etas, jnp.inf)
        
        # Find best available driver
        best_car = jnp.argmin(available_etas)
        eta = etas[best_car]
        
        # Enhanced driver decision: check if drivers will accept this trip
        driver_acceptance = should_drivers_accept_trip(
            eta, current_state, params, driver_decision_key
        )
        
        # The best car must both be available and willing to accept
        driver_will_accept = driver_acceptance[best_car] & online_idle_mask[best_car]
        
        # Customer choice model (existing logic)
        logit = (
            params.w_price * action
            + params.w_intercept  
            + params.w_eta * eta
        )
        p_accept_customer = jnp.exp(logit) / (1 + jnp.exp(logit))
        customer_accept_key, final_step_key = jax.random.split(step_key)
        customer_accepts = jax.random.bernoulli(customer_accept_key, p_accept_customer)
        
        # Trip is fulfilled only if both driver and customer accept
        trip_accepted = driver_will_accept & customer_accepts & (available_etas[best_car] < jnp.inf)
        
        # Revenue calculation
        trip_distance = params.distances[current_state.event.src, current_state.event.dest]
        reward = jax.lax.cond(
            trip_accepted,
            lambda: action * trip_distance,  # Revenue = price per unit × distance
            lambda: 0.0,
        )
        
        # Update driver states and earnings
        updated_state = update_driver_states(
            current_state.event.t, 
            current_state, 
            trip_accepted, 
            jax.lax.select(trip_accepted, best_car, -1), 
            reward
        )
        
        # Update driver locations and times if trip accepted
        def dispatch_driver():
            # Calculate trip duration
            trip_duration = params.distances[updated_state.event.src, updated_state.event.dest]
            trip_end_time = updated_state.event.t + eta + trip_duration
            
            # Update driver location and time
            new_locations = updated_state.locations.at[best_car].set(updated_state.event.dest)
            new_times = updated_state.times.at[best_car].set(trip_end_time)
            
            # Update driver state to ON_TRIP during the trip
            new_driver_states = updated_driver_states.at[best_car].set(DRIVER_STATE_ON_TRIP)
            
            return updated_state.replace(
                locations=new_locations,
                times=new_times, 
                driver_states=new_driver_states
            )
        
        def no_dispatch():
            return updated_state
            
        # Apply dispatch or no dispatch
        dispatched_state = jax.lax.cond(trip_accepted, dispatch_driver, no_dispatch)
        
        # Check for drivers completing trips (times <= current time)        
        current_time = dispatched_state.event.t
        trips_completed = (dispatched_state.times <= current_time) & (dispatched_state.driver_states == DRIVER_STATE_ON_TRIP)
        
        # Update driver states for completed trips
        completed_driver_states = jnp.where(
            trips_completed,
            DRIVER_STATE_ONLINE_IDLE,  # Return to online idle after trip
            dispatched_state.driver_states
        )
        
        # Update trip completion tracking
        new_trips_completed = dispatched_state.trips_completed + trips_completed.astype(int)
        new_time_since_last_trip = jnp.where(
            trips_completed,
            0.0,  # Reset time since last trip
            dispatched_state.time_since_last_trip + 1.0
        )
        
        # Progress to next event
        next_time = dispatched_state.time + 1
        done = next_time >= self.n_events
        
        next_event = jax.lax.cond(
            done,
            lambda: dispatched_state.event,  # Keep current event if done
            lambda: get_nth_event(params.events, next_time)
        )
        
        final_state = dispatched_state.replace(
            time=next_time,
            event=next_event,
            key=final_step_key,
            driver_states=completed_driver_states,
            trips_completed=new_trips_completed,
            time_since_last_trip=new_time_since_last_trip
        )
        
        # Enhanced info with driver dynamics
        online_drivers = jnp.sum(final_state.driver_states == DRIVER_STATE_ONLINE_IDLE)
        avg_earnings = jnp.mean(final_state.daily_earnings)
        
        new_info = {
            "trip_accepted": trip_accepted,
            "customer_p_accept": p_accept_customer,
            "driver_will_accept": driver_will_accept,
            "eta": eta,
            "reward": reward,
            "online_drivers": online_drivers,
            "avg_driver_earnings": avg_earnings,
            "drivers_went_online": jnp.sum(should_go_online),
        }

        return self.get_obs(final_state), final_state, reward, done, new_info

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        return super().reset_env(key, params.dispatch_env_params)


class ManhattanRideshareDispatch(RideshareDispatch):
    def __init__(self, n_cars=10000, n_events=100000):
        super().__init__(n_cars=n_cars, n_nodes=4333, n_events=n_events)

    @property
    def name(self) -> str:
        """Environment name."""
        return "RideshareDispatch-v0"

    @property
    def default_params(self) -> EnvParams:
        # Use local data files instead of remote downloads
        import os
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level to project root
        data_dir = os.path.join(project_root, "data")
        
        events_fname = os.path.join(data_dir, "manhattan-trips.parquet")
        distance_matrix_fname = os.path.join(data_dir, "manhattan-distances.npy")
        
        # Check if files exist
        if not os.path.exists(events_fname):
            raise FileNotFoundError(f"Trip data not found at {events_fname}. Please ensure the data files are in the data/ directory.")
        if not os.path.exists(distance_matrix_fname):
            raise FileNotFoundError(f"Distance data not found at {distance_matrix_fname}. Please ensure the data files are in the data/ directory.")
        
        raw_events = pd.read_parquet(events_fname).head(self.n_events)
        distances_np = np.load(distance_matrix_fname)
        # distances_np[distances_np == 0] = np.inf
        # distances_np = distances_np * (1 - np.eye(distances_np.shape[0]))
        distances = jnp.asarray(np.round(distances_np), dtype=int)

        events = RideshareEvent(
            jnp.asarray(raw_events["t"].values - raw_events["t"].values.min()),
            jnp.asarray(raw_events["pickup_idx"].values),
            jnp.asarray(raw_events["dropoff_idx"].values),
        )
        return EnvParams(events=events, distances=distances, n_cars=self.n_cars)


class ManhattanRidesharePricing(RidesharePricing):
    def __init__(self, n_cars=10000, n_events=100000):
        super().__init__(n_cars=n_cars, n_nodes=4333, n_events=n_events)

    @property
    def name(self) -> str:
        """Environment name."""
        return "RidesharePricing-v0"

    @property
    def default_params(self) -> PricingEnvParams:
        # Use local data files instead of remote downloads
        import os
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level to project root
        data_dir = os.path.join(project_root, "data")
        
        events_fname = os.path.join(data_dir, "manhattan-trips.parquet")
        distance_matrix_fname = os.path.join(data_dir, "manhattan-distances.npy")
        
        # Check if files exist
        if not os.path.exists(events_fname):
            raise FileNotFoundError(f"Trip data not found at {events_fname}. Please ensure the data files are in the data/ directory.")
        if not os.path.exists(distance_matrix_fname):
            raise FileNotFoundError(f"Distance data not found at {distance_matrix_fname}. Please ensure the data files are in the data/ directory.")
        
        raw_events = (
            pd.read_parquet(events_fname).sort_values("t").head(self.n_events)
        )

        # raw_events = pd.read_parquet(events_fname).sort_values("t")
        # # Drop very early years — keep only modern data
        # raw_events = raw_events[raw_events["t"] > 1600000000].head(self.n_events)

        # # 🔍 Add print statements here to inspect timestamps
        # print("Raw timestamps (first 10):", raw_events["t"].values[:10])
        # print("Timestamps after diff:", np.diff(raw_events["t"].values[:10]))

        distances_np = np.load(distance_matrix_fname)
        distances = jnp.asarray(np.round(distances_np), dtype=int)

        events = RideshareEvent(
            jnp.asarray(raw_events["t"].values - raw_events["t"].values.min()),
            jnp.asarray(raw_events["pickup_idx"].values),
            jnp.asarray(raw_events["dropoff_idx"].values),
        )
        
        # Create realistic driver profiles
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducible driver profiles
        keys = jax.random.split(key, 10)
        
        # Experience levels: 25% new, 45% experienced, 30% veteran
        experience_probs = jnp.array([0.25, 0.45, 0.30])
        driver_experiences = jax.random.choice(
            keys[0], 3, (self.n_cars,), p=experience_probs
        )
        
        # Shift patterns: 15% full-time, 35% part-time day, 30% part-time evening, 20% weekend
        shift_probs = jnp.array([0.15, 0.35, 0.30, 0.20])
        driver_shifts = jax.random.choice(
            keys[1], 4, (self.n_cars,), p=shift_probs
        )
        
        # Home locations (use lower-indexed nodes as residential areas)
        residential_nodes = jnp.arange(200, 300)  # Nodes 200-299 as residential
        driver_homes = jax.random.choice(
            keys[2], residential_nodes, (self.n_cars,)
        )
        
        # Daily targets based on shift patterns
        base_hours = jnp.array([10.0, 6.0, 6.0, 8.0])  # hours for each shift type
        base_earnings = jnp.array([200.0, 120.0, 130.0, 150.0])  # earnings for each shift type
        
        daily_hours = base_hours[driver_shifts]
        daily_earnings = base_earnings[driver_shifts]
        
        # Acceptance rates based on experience
        base_acceptance = jnp.array([0.85, 0.75, 0.65])  # acceptance rates by experience
        acceptance_rates = base_acceptance[driver_experiences]
        
        # Multi-platform usage (17% of drivers)
        uses_multi_platform = jax.random.bernoulli(keys[3], 0.17, (self.n_cars,))
        
        enhanced_dispatch_params = EnvParams(
            events=events, 
            distances=distances, 
            n_cars=self.n_cars,
            # Enhanced driver profile parameters
            driver_experience_levels=driver_experiences,
            driver_shift_patterns=driver_shifts,
            driver_home_locations=driver_homes,
            daily_hours_targets=daily_hours,
            daily_earnings_targets=daily_earnings,
            acceptance_rate_baselines=acceptance_rates,
            uses_multiple_platforms=uses_multi_platform,
        )
        
        return PricingEnvParams(
            dispatch_env_params=enhanced_dispatch_params,
            w_price=-0.3,      # More realistic price sensitivity
            w_intercept=4.0,   # Higher base utility
            w_eta=-0.005,      # More realistic ETA sensitivity
        )


@struct.dataclass
class GreedyPolicy(Policy):
    """
    A simple greedy policy that selects the car with the lowest
    estimated time of arrival (ETA) to the pickup location.
    """

    n_cars: int
    temperature: float

    @partial(jax.jit, static_argnums=(0,))
    def apply(
        self,
        env_params: EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        rng, cost_rng = jax.random.split(rng)
        rewards = (
            -self.get_costs(
                env_params, cost_rng, event, locations, times, nn_params
            )
            # Don't dispatch unreachable nodes
            - (env_params.distances[locations, event.src] < 0) * jnp.inf
        )

        action = jax.random.choice(
            rng,
            jnp.arange(self.n_cars),
            p=jnp.exp((rewards - jnp.max(rewards)) / self.temperature),
        )
        return action, {}

    def get_costs(
        self,
        env_params: EnvParams,
        rng: chex.PRNGKey,
        event: RideshareEvent,
        locations: Integer[Array, "n_cars"],
        times: Integer[Array, "n_cars"],
        params: EnvParams,
    ):
        # jax.debug.print(f"{event.t}")
        # jax.debug.print(f"{times}")
        etas = (
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
        )
        return etas

    def init(self, env_params, *args, **kwargs):
        return env_params


@struct.dataclass
class SimplePricingPolicy(Policy):
    n_cars: int
    price_per_distance: float

    def apply(
        self,
        env_params: EnvParams,
        nn_params: Dict,
        obs: Integer[Array, "o_dim"],
        rng: chex.PRNGKey,
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        params = env_params.dispatch_env_params
        # Charge on trip time

        # # Greedy ETD
        # min_etd = jnp.min(
        #     jnp.maximum(event.t, times)
        #     + params.distances[locations, event.src]
        #     + params.distances[event.src, event.dest]
        # )
        return (
            self.price_per_distance * params.distances[event.src, event.dest],
            {},
        )


@struct.dataclass
class ValueGreedyPolicy(GreedyPolicy):
    """
    Uses a value function approximator to compute long term costs,
    then selects the car with the lowest cost.
    """

    nn: nn.Module
    gamma: float

    def obs_to_post_states(
        self, obs: Integer[Array, "o_dim"], env_params: EnvParams
    ):
        event, locations, times = obs_to_state(self.n_cars, obs)
        post_time_deltas = jnp.maximum(
            0,
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
            + env_params.distances[event.src, event.dest]
            - event.t,
        )
        # post_states = jnp.vstack(
        #     [locations, times + post_time_deltas]
        # ).transpose()
        post_states = jnp.vstack(
            [locations, times + post_time_deltas]
        ).transpose()[0]
        return post_states

    def get_costs(
        self,
        env_params: EnvParams,
        rng: chex.PRNGKey,
        event: RideshareEvent,
        locations: Integer[Array, "n_cars"],
        times: Integer[Array, "n_cars"],
        nn_params: Dict,
    ):
        post_time_deltas = jnp.maximum(
            0,
            jnp.maximum(event.t, times)
            + env_params.distances[locations, event.src]
            + env_params.distances[event.src, event.dest]
            - event.t,
        )
        post_states = jnp.vstack(
            [locations, times + post_time_deltas]
        ).transpose()
        # TODO Should use first-order approximation for post-values
        post_values = self.nn.apply(nn_params, post_states[0]).reshape(-1)
        costs = super().get_costs(
            env_params, rng, event, locations, times, nn_params
        )
        return costs - self.gamma * post_values

    def init(self, env_params, rng, obs):
        return self.nn.init(rng, self.obs_to_post_states(obs, env_params))


class RideshareValueNetwork(MLP):
    @classmethod
    def from_env(cls, env, env_params, **kwargs):
        return cls(num_output_units=1, **kwargs)


if __name__ == "__main__":
    n_events = 100
    key = jax.random.PRNGKey(0)
    env = ManhattanRidesharePricing(n_cars=10000, n_events=n_events)
    env_params = env.default_params
    print(env_params)
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.1)
    obs, state = env.reset(key, env_params)
    action, action_info = A.apply(env_params, dict(), obs, key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)

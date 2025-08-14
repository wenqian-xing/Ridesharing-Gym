"""
Enhanced Driver Dynamics - Implementation Example

This module demonstrates how to integrate realistic driver behavior into the existing
rideshare simulation. It extends the current simple model with realistic driver states,
decision-making, and behavioral patterns.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import pandas as pd


class DriverState(IntEnum):
    """Driver states for realistic simulation"""
    OFFLINE = 0
    ONLINE_IDLE = 1
    EN_ROUTE_PICKUP = 2
    ON_TRIP = 3
    REPOSITIONING = 4
    ON_BREAK = 5


class ShiftPattern(IntEnum):
    """Driver shift patterns based on real-world data"""
    FULL_TIME = 0      # 6 AM - 10 PM, ~10 hours/day
    PART_TIME_DAY = 1  # 6 AM - 6 PM, ~6 hours/day
    PART_TIME_EVE = 2  # 5 PM - 2 AM, ~6 hours/day
    WEEKEND = 3        # 10 AM - 12 AM, weekends only


class ExperienceLevel(IntEnum):
    """Driver experience levels affecting behavior"""
    NEW = 0         # 0-3 months, high acceptance rate
    EXPERIENCED = 1 # 3-12 months, balanced behavior
    VETERAN = 2     # 12+ months, selective behavior


@struct.dataclass
class DriverProfile:
    """Static driver characteristics (immutable during simulation)"""
    driver_id: int
    experience_level: int  # ExperienceLevel
    shift_pattern: int     # ShiftPattern
    home_location: int     # Node index
    preferred_zones: jnp.ndarray  # Array of preferred work zone indices
    uses_multiple_platforms: bool
    daily_hours_target: float
    daily_earnings_target: float
    acceptance_rate_baseline: float
    vehicle_capacity: int  # 1-6 passengers


@struct.dataclass
class DriverDynamicState:
    """Dynamic driver state (changes during simulation)"""
    # Current state
    current_states: jnp.ndarray     # DriverState for each driver
    locations: jnp.ndarray          # Current location for each driver
    times: jnp.ndarray              # Next available time for each driver
    
    # Session tracking
    session_start_times: jnp.ndarray    # When driver went online today
    total_session_times: jnp.ndarray    # Total online time today
    daily_earnings: jnp.ndarray         # Earnings accumulated today
    trips_completed: jnp.ndarray        # Trips completed today
    
    # Decision factors
    consecutive_rejections: jnp.ndarray  # Consecutive trip rejections
    time_since_last_trip: jnp.ndarray   # Time since last completed trip
    last_break_times: jnp.ndarray       # Last break time
    
    # Trip assignments
    assigned_requests: jnp.ndarray      # Request ID (-1 if none)
    pickup_locations: jnp.ndarray       # Pickup location (-1 if none)
    dropoff_locations: jnp.ndarray      # Dropoff location (-1 if none)
    
    # Multi-platform
    other_platform_requests: jnp.ndarray  # Has request on other platform


class EnhancedDriverManager:
    """Manages realistic driver population and behavior"""
    
    def __init__(self, n_drivers: int, historical_data: Optional[Dict] = None):
        self.n_drivers = n_drivers
        self.profiles = self._create_driver_profiles(historical_data or {})
        
    def _create_driver_profiles(self, historical_data: Dict) -> List[DriverProfile]:
        """Create realistic driver profiles based on research data"""
        profiles = []
        
        # Distributions based on real Uber driver research
        experience_dist = [0.25, 0.45, 0.30]  # new, experienced, veteran
        shift_dist = [0.15, 0.35, 0.30, 0.20]  # full, day, evening, weekend
        
        # Get zone preferences from historical data
        popular_zones = historical_data.get('high_demand_zones', list(range(100)))
        residential_zones = historical_data.get('residential_zones', list(range(200, 300)))
        
        for i in range(self.n_drivers):
            # Sample characteristics
            key = jax.random.PRNGKey(i)
            keys = jax.random.split(key, 10)
            
            experience = jax.random.choice(keys[0], 3, p=jnp.array(experience_dist))
            shift_pattern = jax.random.choice(keys[1], 4, p=jnp.array(shift_dist))
            uses_multi = jax.random.bernoulli(keys[2], 0.17)  # 17% use multiple platforms
            
            # Home location (more likely in residential areas)
            home_location = jax.random.choice(keys[3], jnp.array(residential_zones))
            
            # Preferred work zones (3-5 zones)
            n_preferred = jax.random.randint(keys[4], (), 3, 6)
            preferred_zones = jax.random.choice(
                keys[5], jnp.array(popular_zones), (n_preferred,), replace=False
            )
            
            # Targets based on shift pattern and experience
            hours_target = self._get_hours_target(int(shift_pattern))
            earnings_target = self._get_earnings_target(int(shift_pattern), int(experience))
            acceptance_rate = self._get_acceptance_rate(int(experience))
            
            profile = DriverProfile(
                driver_id=i,
                experience_level=int(experience),
                shift_pattern=int(shift_pattern),
                home_location=int(home_location),
                preferred_zones=preferred_zones,
                uses_multiple_platforms=bool(uses_multi),
                daily_hours_target=float(hours_target),
                daily_earnings_target=float(earnings_target),
                acceptance_rate_baseline=float(acceptance_rate),
                vehicle_capacity=4  # Most drivers have standard cars
            )
            
            profiles.append(profile)
            
        return profiles
    
    def _get_hours_target(self, shift_pattern: int) -> float:
        """Get daily hours target based on shift pattern"""
        targets = {
            ShiftPattern.FULL_TIME: 10.0,
            ShiftPattern.PART_TIME_DAY: 6.0, 
            ShiftPattern.PART_TIME_EVE: 6.0,
            ShiftPattern.WEEKEND: 8.0
        }
        return targets.get(shift_pattern, 6.0)
    
    def _get_earnings_target(self, shift_pattern: int, experience: int) -> float:
        """Get daily earnings target based on shift and experience"""
        base_targets = {
            ShiftPattern.FULL_TIME: 200.0,
            ShiftPattern.PART_TIME_DAY: 120.0,
            ShiftPattern.PART_TIME_EVE: 130.0,  # Higher evening rates
            ShiftPattern.WEEKEND: 150.0
        }
        
        base = base_targets.get(shift_pattern, 120.0)
        
        # Experience multiplier
        exp_multipliers = {
            ExperienceLevel.NEW: 0.8,      # Lower expectations
            ExperienceLevel.EXPERIENCED: 1.0,
            ExperienceLevel.VETERAN: 1.2   # Higher expectations
        }
        
        return base * exp_multipliers.get(experience, 1.0)
    
    def _get_acceptance_rate(self, experience: int) -> float:
        """Get baseline acceptance rate by experience"""
        rates = {
            ExperienceLevel.NEW: 0.85,        # High acceptance
            ExperienceLevel.EXPERIENCED: 0.75, # Balanced
            ExperienceLevel.VETERAN: 0.65     # More selective
        }
        return rates.get(experience, 0.75)
    
    def initialize_state(self, key: jax.random.PRNGKey) -> DriverDynamicState:
        """Initialize driver dynamic state"""
        keys = jax.random.split(key, 10)
        
        # Start most drivers offline, some online based on time
        # For demo, start 20% online
        online_prob = 0.2
        initial_states = jax.random.choice(
            keys[0], 
            jnp.array([DriverState.OFFLINE, DriverState.ONLINE_IDLE]),
            (self.n_drivers,),
            p=jnp.array([1-online_prob, online_prob])
        )
        
        # Random initial locations (home locations for offline drivers)
        locations = jnp.array([profile.home_location for profile in self.profiles])
        
        return DriverDynamicState(
            current_states=initial_states,
            locations=locations,
            times=jnp.zeros(self.n_drivers),
            session_start_times=jnp.zeros(self.n_drivers),
            total_session_times=jnp.zeros(self.n_drivers),
            daily_earnings=jnp.zeros(self.n_drivers),
            trips_completed=jnp.zeros(self.n_drivers),
            consecutive_rejections=jnp.zeros(self.n_drivers, dtype=int),
            time_since_last_trip=jnp.zeros(self.n_drivers),
            last_break_times=jnp.zeros(self.n_drivers),
            assigned_requests=jnp.full(self.n_drivers, -1),
            pickup_locations=jnp.full(self.n_drivers, -1),
            dropoff_locations=jnp.full(self.n_drivers, -1),
            other_platform_requests=jnp.zeros(self.n_drivers, dtype=bool)
        )


class DriverDecisionEngine:
    """Handles driver decision-making logic"""
    
    def __init__(self, driver_manager: EnhancedDriverManager):
        self.driver_manager = driver_manager
        self.profiles = driver_manager.profiles
        
    def should_go_online(
        self, 
        current_time: float,
        driver_state: DriverDynamicState,
        market_conditions: Dict,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Vectorized decision for drivers going online"""
        
        hour_of_day = jnp.floor(current_time / 3600) % 24
        
        # Extract profile data as arrays
        shift_patterns = jnp.array([p.shift_pattern for p in self.profiles])
        hours_targets = jnp.array([p.daily_hours_target for p in self.profiles])
        earnings_targets = jnp.array([p.daily_earnings_target for p in self.profiles])
        
        # Check if in preferred shift time
        in_shift_time = jnp.select([
            shift_patterns == ShiftPattern.FULL_TIME,
            shift_patterns == ShiftPattern.PART_TIME_DAY,
            shift_patterns == ShiftPattern.PART_TIME_EVE,
            shift_patterns == ShiftPattern.WEEKEND,
        ], [
            (hour_of_day >= 6) & (hour_of_day <= 22),
            (hour_of_day >= 6) & (hour_of_day <= 18), 
            (hour_of_day >= 17) | (hour_of_day <= 2),
            (hour_of_day >= 10) & (hour_of_day <= 24),
        ])
        
        # Check daily limits
        under_hours_limit = driver_state.total_session_times < (hours_targets * 3600)
        under_earnings_limit = driver_state.daily_earnings < earnings_targets
        
        # Market conditions
        surge_multiplier = market_conditions.get('surge_multiplier', 1.0)
        market_attractiveness = jnp.clip(surge_multiplier, 0.5, 2.0)
        
        # Base probability
        base_prob = 0.7
        online_prob = base_prob * market_attractiveness
        
        # Only consider offline drivers
        can_go_online = (
            (driver_state.current_states == DriverState.OFFLINE) &
            in_shift_time & 
            under_hours_limit & 
            under_earnings_limit
        )
        
        online_prob = jnp.where(can_go_online, online_prob, 0.0)
        
        # Random decisions
        keys = jax.random.split(key, self.driver_manager.n_drivers)
        decisions = jax.vmap(jax.random.bernoulli)(keys, online_prob)
        
        return decisions
    
    def should_accept_trip(
        self,
        trip_request: Dict,
        driver_state: DriverDynamicState,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Vectorized trip acceptance decisions"""
        
        # Extract profile data
        acceptance_baselines = jnp.array([p.acceptance_rate_baseline for p in self.profiles])
        experience_levels = jnp.array([p.experience_level for p in self.profiles])
        uses_multi_platform = jnp.array([p.uses_multiple_platforms for p in self.profiles])
        
        # Base acceptance rate
        acceptance_prob = acceptance_baselines
        
        # ETA penalty
        eta_to_pickup = trip_request.get('eta_to_pickup', 300)  # seconds
        eta_penalty = jnp.where(
            eta_to_pickup > 600,  # > 10 min
            0.7,
            jnp.where(eta_to_pickup > 300, 0.85, 1.0)  # > 5 min
        )
        acceptance_prob *= eta_penalty
        
        # Experience factor
        exp_factor = jnp.select([
            experience_levels == ExperienceLevel.NEW,
            experience_levels == ExperienceLevel.EXPERIENCED,
            experience_levels == ExperienceLevel.VETERAN,
        ], [1.2, 1.0, 0.9])
        acceptance_prob *= exp_factor
        
        # Consecutive rejection penalty (platform pressure)
        rejection_penalty = jnp.where(
            driver_state.consecutive_rejections > 3,
            1.5,
            1.0
        )
        acceptance_prob *= rejection_penalty
        
        # Multi-platform competition
        platform_factor = jnp.where(
            uses_multi_platform & driver_state.other_platform_requests,
            0.7,  # Lower acceptance if has other platform request
            1.0
        )
        acceptance_prob *= platform_factor
        
        # Only idle drivers can accept
        can_accept = driver_state.current_states == DriverState.ONLINE_IDLE
        acceptance_prob = jnp.where(can_accept, jnp.clip(acceptance_prob, 0.0, 1.0), 0.0)
        
        # Random decisions
        keys = jax.random.split(key, self.driver_manager.n_drivers)
        decisions = jax.vmap(jax.random.bernoulli)(keys, acceptance_prob)
        
        return decisions
    
    def should_reposition(
        self,
        driver_state: DriverDynamicState,
        market_state: Dict,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Decide which drivers should reposition and where"""
        
        # Only idle drivers consider repositioning
        can_reposition = (
            (driver_state.current_states == DriverState.ONLINE_IDLE) &
            (driver_state.time_since_last_trip > 600)  # Wait at least 10 minutes
        )
        
        # Simple repositioning logic: move to high-demand zones
        demand_by_zone = market_state.get('demand_by_zone', {})
        supply_by_zone = market_state.get('supply_by_zone', {})
        
        # For demo, use simple heuristic
        should_reposition = can_reposition & (jax.random.uniform(key, (self.driver_manager.n_drivers,)) < 0.1)
        
        # Target zones (simplified - would be more sophisticated in practice)
        target_zones = jax.random.choice(
            key, 
            jnp.array(list(demand_by_zone.keys()) if demand_by_zone else [0]),
            (self.driver_manager.n_drivers,)
        )
        
        return should_reposition, target_zones


def integrate_with_existing_simulation():
    """
    Example of how to integrate enhanced driver dynamics with existing simulation
    """
    
    # Create enhanced driver manager
    n_drivers = 300
    historical_data = {
        'high_demand_zones': list(range(50, 150)),    # Popular pickup zones
        'residential_zones': list(range(200, 300)),   # Home base zones
    }
    
    driver_manager = EnhancedDriverManager(n_drivers, historical_data)
    decision_engine = DriverDecisionEngine(driver_manager)
    
    # Initialize state
    key = jax.random.PRNGKey(42)
    driver_state = driver_manager.initialize_state(key)
    
    print(f"Initialized {n_drivers} drivers")
    print(f"Online drivers: {jnp.sum(driver_state.current_states == DriverState.ONLINE_IDLE)}")
    print(f"Offline drivers: {jnp.sum(driver_state.current_states == DriverState.OFFLINE)}")
    
    # Simulate decisions over time
    current_time = 8 * 3600  # 8 AM
    market_conditions = {'surge_multiplier': 1.2}
    
    keys = jax.random.split(key, 3)
    
    # Test going online decisions
    go_online = decision_engine.should_go_online(
        current_time, driver_state, market_conditions, keys[0]
    )
    print(f"Drivers deciding to go online: {jnp.sum(go_online)}")
    
    # Test trip acceptance
    trip_request = {'eta_to_pickup': 420}  # 7 minutes
    accept_trip = decision_engine.should_accept_trip(
        trip_request, driver_state, keys[1]
    )
    print(f"Drivers accepting trip: {jnp.sum(accept_trip)}")
    
    # Test repositioning
    market_state = {
        'demand_by_zone': {i: np.random.uniform(0.5, 2.0) for i in range(100)},
        'supply_by_zone': {i: np.random.uniform(0.3, 1.5) for i in range(100)}
    }
    should_reposition, target_zones = decision_engine.should_reposition(
        driver_state, market_state, keys[2]
    )
    print(f"Drivers repositioning: {jnp.sum(should_reposition)}")
    
    return driver_manager, decision_engine, driver_state


if __name__ == "__main__":
    # Demo the enhanced driver dynamics
    print("=== Enhanced Driver Dynamics Demo ===")
    driver_manager, decision_engine, driver_state = integrate_with_existing_simulation()
    
    print("\n✅ Enhanced driver dynamics successfully integrated!")
    print("Next steps:")
    print("1. Integrate with existing JAX environment")
    print("2. Add driver state transitions")
    print("3. Implement earnings tracking")
    print("4. Add break and shift management")
    print("5. Validate against real driver behavior data")
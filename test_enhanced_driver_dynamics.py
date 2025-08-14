#!/usr/bin/env python3
"""
Test script for enhanced driver dynamics integration
"""

import sys
import os
sys.path.append('.')

import jax
import jax.numpy as jnp
import numpy as np
from picard.rideshare_dispatch import ManhattanRidesharePricing

def test_enhanced_driver_dynamics():
    """Test that enhanced driver dynamics work in the simulation"""
    print("=== Testing Enhanced Driver Dynamics Integration ===")
    
    # Create environment with smaller scale for testing
    env = ManhattanRidesharePricing(n_cars=100, n_events=1000)
    
    # Get default parameters
    try:
        params = env.default_params
        print(f"✅ Successfully loaded parameters with {params.n_cars} drivers")
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test environment reset
    try:
        key = jax.random.PRNGKey(42)
        obs, state = env.reset_env(key, params)
        print(f"✅ Environment reset successful")
        print(f"   - Driver states shape: {state.driver_states.shape}")
        print(f"   - Initial online drivers: {jnp.sum(state.driver_states == 1)}")  # ONLINE_IDLE = 1
        print(f"   - Initial offline drivers: {jnp.sum(state.driver_states == 0)}")  # OFFLINE = 0
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return False
    
    # Test a few steps
    try:
        print("\n--- Testing Environment Steps ---")
        
        for step in range(5):
            # Random pricing action
            action = jax.random.uniform(jax.random.PRNGKey(step + 100), (), minval=1.0, maxval=3.0)
            
            # Step environment
            step_key = jax.random.PRNGKey(step + 200)
            obs, state, reward, done, info = env.step_env(step_key, state, action, params)
            
            print(f"Step {step + 1}:")
            print(f"   - Action (price): {action:.2f}")
            print(f"   - Reward: {reward:.2f}")
            print(f"   - Trip accepted: {info.get('trip_accepted', False)}")
            print(f"   - Customer P accept: {info.get('customer_p_accept', 0):.3f}")
            print(f"   - Driver will accept: {info.get('driver_will_accept', False)}")
            print(f"   - ETA: {info.get('eta', 0):.1f}")
            print(f"   - Online drivers: {info.get('online_drivers', 0)}")
            print(f"   - Drivers went online: {info.get('drivers_went_online', 0)}")
            print(f"   - Avg earnings: ${info.get('avg_driver_earnings', 0):.2f}")
            
            if done:
                print(f"   - Environment done at step {step + 1}")
                break
                
        print("✅ Environment steps completed successfully")
        
    except Exception as e:
        print(f"❌ Error during environment steps: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test driver state distribution
    try:
        print("\n--- Driver State Analysis ---")
        offline = jnp.sum(state.driver_states == 0)
        online_idle = jnp.sum(state.driver_states == 1)
        on_trip = jnp.sum(state.driver_states == 3)
        
        print(f"   - Offline drivers: {offline} ({offline/params.n_cars*100:.1f}%)")
        print(f"   - Online idle: {online_idle} ({online_idle/params.n_cars*100:.1f}%)")
        print(f"   - On trip: {on_trip} ({on_trip/params.n_cars*100:.1f}%)")
        print(f"   - Total trips completed: {jnp.sum(state.trips_completed)}")
        print(f"   - Total earnings: ${jnp.sum(state.daily_earnings):.2f}")
        
        print("✅ Driver state analysis completed")
        
    except Exception as e:
        print(f"❌ Error in driver state analysis: {e}")
        return False
    
    print("\n🎉 All tests passed! Enhanced driver dynamics integration successful!")
    return True

if __name__ == "__main__":
    success = test_enhanced_driver_dynamics()
    sys.exit(0 if success else 1)
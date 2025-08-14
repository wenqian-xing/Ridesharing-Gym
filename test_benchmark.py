"""
Test suite for the Treatment Effect Estimation Benchmark.

This module provides basic validation and testing for the benchmark components.
"""

import numpy as np
import pytest
from typing import Tuple, Any

from treatment_effect_gym import TreatmentBenchmark, BenchmarkConfig
from environments.two_state_mdp import TwoStateMDPEnvironment
from environments.queueing import QueueingEnvironment
from experiment_designs.randomized import RandomizedDesign
from experiment_designs.switchback import SwitchbackDesign
from estimators.naive import NaiveEstimator
from estimators.truncated_dq import TruncatedDQEstimator
from estimators.dq import DQEstimator


class TestTwoStateMDPEnvironment:
    """Test the two-state MDP environment."""
    
    def setup_method(self):
        """Setup for each test."""
        self.env = TwoStateMDPEnvironment(
            mixing_coeff=0.5,
            treatment_bias=0.1,
            smoothness=0.5,
            noise_std=0.02,
            reward_std=0.1
        )
    
    def test_environment_creation(self):
        """Test environment can be created."""
        assert self.env.name == "TwoStateMDP"
        assert self.env.mixing_coeff == 0.5
        assert self.env.treatment_bias == 0.1
    
    def test_treatment_simulation(self):
        """Test treatment policy simulation."""
        T = 100
        rewards, actions, states = self.env.simulate_treatment_policy(T, seed=42)
        
        assert len(rewards) == T
        assert len(actions) == T
        assert len(states) == T
        assert np.all(actions == 1)  # All treatment
        assert np.all(np.isin(states, [0, 1]))  # Valid states
    
    def test_control_simulation(self):
        """Test control policy simulation."""
        T = 100
        rewards, actions, states = self.env.simulate_control_policy(T, seed=42)
        
        assert len(rewards) == T
        assert len(actions) == T
        assert len(states) == T
        assert np.all(actions == 0)  # All control
        assert np.all(np.isin(states, [0, 1]))  # Valid states
    
    def test_true_ate_computation(self):
        """Test true ATE computation."""
        true_ate = self.env.compute_true_ate(T=1000, seed=42)
        assert isinstance(true_ate, float)
        assert not np.isnan(true_ate)
        assert not np.isinf(true_ate)
    
    def test_experiment_simulation(self):
        """Test experiment simulation with design."""
        design = RandomizedDesign(p=0.5)
        T = 100
        
        rewards, actions, states = self.env.simulate_experiment(design, T, seed=42)
        
        assert len(rewards) == T
        assert len(actions) == T
        assert len(states) == T
        assert np.all(np.isin(actions, [0, 1]))  # Valid actions
        assert np.all(np.isin(states, [0, 1]))  # Valid states
        
        # Should have mix of treatment and control
        assert np.sum(actions == 1) > 0
        assert np.sum(actions == 0) > 0


class TestExperimentDesigns:
    """Test experimental designs."""
    
    def test_randomized_design(self):
        """Test randomized A/B design."""
        design = RandomizedDesign(p=0.5)
        design.reset(seed=42)
        
        # Generate assignments
        assignments = [design.assign_treatment(t, 0) for t in range(100)]
        
        assert all(a in [0, 1] for a in assignments)
        
        # Should be roughly balanced
        treatment_rate = np.mean(assignments)
        assert 0.3 < treatment_rate < 0.7  # Allow some variance
    
    def test_switchback_design(self):
        """Test switchback design."""
        design = SwitchbackDesign(switch_every=10, p=0.5)
        design.reset(seed=42)
        
        # Generate assignments
        assignments = [design.assign_treatment(t, 0, max_time=100) for t in range(50)]
        
        assert all(a in [0, 1] for a in assignments)
        
        # Should switch in blocks
        # First 10 should be same
        first_block = assignments[:10]
        assert len(set(first_block)) == 1
        
        # Next 10 might be different
        second_block = assignments[10:20]
        assert len(set(second_block)) == 1
    
    def test_design_reset(self):
        """Test design reset functionality."""
        design = RandomizedDesign(p=0.5)
        
        # Same seed should give same sequence
        design.reset(seed=42)
        seq1 = [design.assign_treatment(t, 0) for t in range(20)]
        
        design.reset(seed=42)
        seq2 = [design.assign_treatment(t, 0) for t in range(20)]
        
        assert seq1 == seq2


class TestEstimators:
    """Test treatment effect estimators."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.T = 100
        
        # Generate synthetic data
        self.rewards = np.random.normal(2.0, 1.0, self.T)
        self.actions = np.random.choice([0, 1], self.T)
        self.states = np.random.randint(0, 2, (self.T, 1))
        
        # Add treatment effect
        treatment_effect = 0.5
        self.rewards[self.actions == 1] += treatment_effect
    
    def test_naive_estimator(self):
        """Test naive difference-in-means estimator."""
        estimator = NaiveEstimator(discount_factor=1.0)
        estimate = estimator.estimate(self.rewards, self.actions, self.states)
        
        assert isinstance(estimate, float)
        assert not np.isnan(estimate)
        assert not np.isinf(estimate)
        
        # Should be roughly correct (within reasonable range)
        assert -2.0 < estimate < 2.0
    
    def test_dq_estimator(self):
        """Test DQ estimator."""
        estimator = DQEstimator(discount_factor=1.0)
        estimate = estimator.estimate(self.rewards, self.actions, self.states)
        
        assert isinstance(estimate, float)
        assert not np.isnan(estimate)
        assert not np.isinf(estimate)
    
    def test_truncated_dq_estimator(self):
        """Test truncated DQ estimator."""
        for k in [1, 3, 5, 10]:
            estimator = TruncatedDQEstimator(k=k, discount_factor=1.0)
            estimate = estimator.estimate(self.rewards, self.actions, self.states)
            
            assert isinstance(estimate, float)
            assert not np.isnan(estimate)
            assert not np.isinf(estimate)
    
    def test_estimator_edge_cases(self):
        """Test estimator edge cases."""
        estimator = NaiveEstimator()
        
        # All treatment
        all_treatment_actions = np.ones(10)
        rewards = np.random.normal(0, 1, 10)
        states = np.random.normal(0, 1, (10, 1))
        
        estimate = estimator.estimate(rewards, all_treatment_actions, states)
        assert estimate == 0.0  # Should handle gracefully
        
        # All control
        all_control_actions = np.zeros(10)
        estimate = estimator.estimate(rewards, all_control_actions, states)
        assert estimate == 0.0  # Should handle gracefully
    
    def test_estimator_names(self):
        """Test estimator names are set correctly."""
        estimators = [
            NaiveEstimator(),
            DQEstimator(), 
            TruncatedDQEstimator(k=3)
        ]
        
        for est in estimators:
            assert hasattr(est, 'name')
            assert isinstance(est.name, str)
            assert len(est.name) > 0


class TestBenchmark:
    """Test the main benchmark system."""
    
    def setup_method(self):
        """Setup benchmark for testing."""
        self.benchmark = TreatmentBenchmark("test_results")
        
        # Register minimal components
        self.benchmark.register_environment("test_mdp", TwoStateMDPEnvironment())
        self.benchmark.register_design("test_ab", RandomizedDesign(p=0.5))
        self.benchmark.register_estimator("test_naive", NaiveEstimator())
    
    def test_benchmark_creation(self):
        """Test benchmark can be created."""
        assert self.benchmark.output_dir.name == "test_results"
        assert len(self.benchmark.environments) == 1
        assert len(self.benchmark.designs) == 1
        assert len(self.benchmark.estimators) == 1
    
    def test_component_registration(self):
        """Test component registration."""
        components = self.benchmark.list_components()
        
        assert "test_mdp" in components['environments']
        assert "test_ab" in components['designs'] 
        assert "test_naive" in components['estimators']
    
    def test_single_run(self):
        """Test single benchmark run."""
        result = self.benchmark.run(
            environment="test_mdp",
            experiment_design="test_ab",
            estimator="test_naive",
            n_trials=5,  # Small for testing
            T=50,
            seed=42
        )
        
        assert result.config.environment == "test_mdp"
        assert result.config.experiment_design == "test_ab"
        assert result.config.estimator == "test_naive"
        assert len(result.estimates) == 5
        assert isinstance(result.true_ate, float)
        assert result.success_rate >= 0.0
        assert result.execution_time > 0.0
    
    def test_benchmark_suite(self):
        """Test benchmark suite execution."""
        configs = [
            BenchmarkConfig(
                environment="test_mdp",
                experiment_design="test_ab", 
                estimator="test_naive",
                n_trials=3,
                seed=42
            )
        ]
        
        results = self.benchmark.run_suite(configs)
        
        assert len(results) == 1
        result_key = list(results.keys())[0]
        assert "test_mdp" in result_key
        assert "test_ab" in result_key
        assert "test_naive" in result_key
    
    def test_result_summary_stats(self):
        """Test result summary statistics."""
        result = self.benchmark.run(
            environment="test_mdp",
            experiment_design="test_ab",
            estimator="test_naive", 
            n_trials=5,
            T=50,
            seed=42
        )
        
        stats = result.summary_stats()
        
        required_keys = ['true_ate', 'mean_estimate', 'bias', 'abs_bias', 'mse', 'std', 'success_rate']
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert not np.isnan(stats[key])


class TestIntegration:
    """Integration tests for the full system."""
    
    def test_full_workflow(self):
        """Test complete benchmark workflow."""
        # Setup
        benchmark = TreatmentBenchmark("integration_test")
        
        env = TwoStateMDPEnvironment()
        design = RandomizedDesign(p=0.5)
        estimator = TruncatedDQEstimator(k=3)
        
        benchmark.register_environment("mdp", env)
        benchmark.register_design("ab", design)
        benchmark.register_estimator("truncated", estimator)
        
        # Run
        result = benchmark.run(
            environment="mdp",
            experiment_design="ab",
            estimator="truncated",
            n_trials=10,
            T=100,
            seed=42
        )
        
        # Validate
        assert result.success_rate > 0.8  # Most trials should succeed
        assert len(result.estimates) == 10
        stats = result.summary_stats()
        assert abs(stats['bias']) < 5.0  # Reasonable bias
        assert stats['mse'] > 0.0  # Non-zero MSE
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        benchmark = TreatmentBenchmark("repro_test")
        
        benchmark.register_environment("mdp", TwoStateMDPEnvironment())
        benchmark.register_design("ab", RandomizedDesign(p=0.5))
        benchmark.register_estimator("naive", NaiveEstimator())
        
        # Run twice with same seed
        result1 = benchmark.run("mdp", "ab", "naive", n_trials=5, T=50, seed=123)
        result2 = benchmark.run("mdp", "ab", "naive", n_trials=5, T=50, seed=123)
        
        # Should get identical results
        assert result1.true_ate == result2.true_ate
        assert np.allclose(result1.estimates, result2.estimates)


def run_basic_validation():
    """Run basic validation checks."""
    print("Running basic validation...")
    
    # Test environment
    env = TwoStateMDPEnvironment()
    true_ate = env.compute_true_ate(T=1000, seed=42)
    print(f"✓ Environment creates valid ATE: {true_ate:.4f}")
    
    # Test design
    design = RandomizedDesign(p=0.5)
    design.reset(seed=42)
    assignments = [design.assign_treatment(t, 0) for t in range(100)]
    treatment_rate = np.mean(assignments)
    print(f"✓ Design assigns treatments: {treatment_rate:.2%} treatment rate")
    
    # Test estimator
    np.random.seed(42)
    rewards = np.random.normal(2.0, 1.0, 100)
    actions = np.random.choice([0, 1], 100)
    states = np.random.randint(0, 2, (100, 1))
    
    estimator = TruncatedDQEstimator(k=3)
    estimate = estimator.estimate(rewards, actions, states)
    print(f"✓ Estimator produces estimate: {estimate:.4f}")
    
    # Test benchmark
    benchmark = TreatmentBenchmark("validation_test")
    benchmark.register_environment("mdp", env)
    benchmark.register_design("ab", design)
    benchmark.register_estimator("truncated", estimator)
    
    result = benchmark.run("mdp", "ab", "truncated", n_trials=5, T=50, seed=42)
    print(f"✓ Benchmark runs successfully: {result.success_rate:.1%} success rate")
    
    print("All validation checks passed!")


if __name__ == "__main__":
    # Run basic validation
    run_basic_validation()
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, skipping detailed tests")
        print("Install with: pip install pytest")
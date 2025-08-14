"""
Treatment Effect Estimation Benchmark - A Gym-like Interface

This module provides a unified interface for benchmarking treatment effect estimation
methods across different environments and experimental designs, similar to OpenAI Gym.

Core Components:
- TreatmentEnvironment: Simulation environments (MDP, queueing, rideshare)
- ExperimentDesign: Data collection policies (switchback, A/B testing)
- TreatmentEstimator: Estimation algorithms (DM, DQ, truncated DQ, etc.)
- TreatmentBenchmark: Main interface for running benchmarks

Usage:
    benchmark = TreatmentBenchmark()
    results = benchmark.run(
        environment="queueing",
        experiment_design="switchback", 
        estimator="truncated_dq",
        n_trials=100
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    environment: str
    experiment_design: str
    estimator: str
    n_trials: int = 100
    seed: int = 42
    environment_params: Dict[str, Any] = field(default_factory=dict)
    design_params: Dict[str, Any] = field(default_factory=dict)
    estimator_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a benchmark experiment."""
    config: BenchmarkConfig
    true_ate: float
    estimates: List[float]
    execution_time: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics."""
        estimates = np.array(self.estimates)
        bias = estimates - self.true_ate
        return {
            'true_ate': self.true_ate,
            'mean_estimate': np.mean(estimates),
            'bias': np.mean(bias),
            'abs_bias': np.mean(np.abs(bias)),
            'mse': np.mean(bias**2),
            'std': np.std(estimates),
            'success_rate': self.success_rate
        }


class TreatmentEnvironment(ABC):
    """Abstract base class for treatment effect simulation environments."""
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment state."""
        pass
    
    @abstractmethod
    def simulate_treatment_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure treatment policy.
        
        Returns:
            rewards, actions, states
        """
        pass
    
    @abstractmethod
    def simulate_control_policy(self, T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate pure control policy.
        
        Returns:
            rewards, actions, states
        """
        pass
    
    @abstractmethod
    def simulate_experiment(self, design: 'ExperimentDesign', T: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate experiment according to design.
        
        Returns:
            rewards, actions, states
        """
        pass
    
    @abstractmethod
    def compute_true_ate(self, T: int, seed: Optional[int] = None) -> float:
        """Compute true average treatment effect."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name."""
        pass


class ExperimentDesign(ABC):
    """Abstract base class for experimental designs."""
    
    @abstractmethod
    def assign_treatment(self, t: int, state: Any, **kwargs) -> int:
        """Assign treatment at time t given state.
        
        Returns:
            action: 0 for control, 1 for treatment
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset design state."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Design name."""
        pass


class TreatmentEstimator(ABC):
    """Abstract base class for treatment effect estimators."""
    
    @abstractmethod
    def estimate(self, rewards: np.ndarray, actions: np.ndarray, states: np.ndarray, **kwargs) -> float:
        """Estimate treatment effect from trajectory data.
        
        Args:
            rewards: (T,) array of rewards
            actions: (T,) array of actions (0=control, 1=treatment)  
            states: (T, d) array of states
            
        Returns:
            treatment_effect_estimate: float
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Estimator name."""
        pass


class TreatmentBenchmark:
    """Main benchmark interface for running treatment effect estimation experiments."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Registry of components
        self.environments: Dict[str, TreatmentEnvironment] = {}
        self.designs: Dict[str, ExperimentDesign] = {}
        self.estimators: Dict[str, TreatmentEstimator] = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup benchmark logging."""
        log_file = self.output_dir / "benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def register_environment(self, name: str, env: TreatmentEnvironment):
        """Register a treatment environment."""
        self.environments[name] = env
        self.logger.info(f"Registered environment: {name}")
    
    def register_design(self, name: str, design: ExperimentDesign):
        """Register an experimental design."""
        self.designs[name] = design
        self.logger.info(f"Registered design: {name}")
    
    def register_estimator(self, name: str, estimator: TreatmentEstimator):
        """Register a treatment effect estimator."""
        self.estimators[name] = estimator
        self.logger.info(f"Registered estimator: {name}")
    
    def run(self, 
            environment: str,
            experiment_design: str, 
            estimator: str,
            n_trials: int = 100,
            T: int = 1000,
            seed: int = 42,
            **kwargs) -> BenchmarkResult:
        """Run benchmark experiment.
        
        Args:
            environment: Name of registered environment
            experiment_design: Name of registered design
            estimator: Name of registered estimator
            n_trials: Number of trials to run
            T: Length of each trial
            seed: Random seed
            **kwargs: Additional parameters for components
            
        Returns:
            BenchmarkResult with estimates and statistics
        """
        
        # Get components
        env = self.environments[environment]
        design = self.designs[experiment_design]
        est = self.estimators[estimator]
        
        self.logger.info(f"Starting benchmark: {environment} + {experiment_design} + {estimator}")
        start_time = datetime.now()
        
        # Compute true ATE
        true_ate = env.compute_true_ate(T, seed)
        
        estimates = []
        failed_trials = 0
        
        # Check if environment supports batch simulation for performance
        if hasattr(env, 'simulate_batch_experiments'):
            self.logger.info(f"Using batch simulation for {n_trials} trials")
            try:
                # Use batch simulation for improved performance
                batch_kwargs = {k: v for k, v in kwargs.items() if k not in ['T', 'seed']}
                all_rewards, all_actions, all_states = env.simulate_batch_experiments(
                    design, T=T, n_trials=n_trials, seed=seed, **batch_kwargs
                )
                
                # Process batch results
                for trial in range(len(all_rewards)):
                    try:
                        rewards = all_rewards[trial]
                        actions = all_actions[trial]
                        states = all_states[trial]
                        
                        # Estimate treatment effect
                        estimate = est.estimate(rewards, actions, states, **kwargs)
                        estimates.append(estimate)
                        
                    except Exception as e:
                        self.logger.warning(f"Trial {trial} estimation failed: {str(e)}")
                        failed_trials += 1
                        
            except Exception as e:
                self.logger.warning(f"Batch simulation failed, falling back to individual trials: {str(e)}")
                # Fall back to individual trials
                estimates, failed_trials = self._run_individual_trials(env, design, est, n_trials, T, seed, **kwargs)
        else:
            # Fall back to individual trials for environments without batch support
            self.logger.info(f"Using individual trial simulation for {n_trials} trials")
            estimates, failed_trials = self._run_individual_trials(env, design, est, n_trials, T, seed, **kwargs)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        success_rate = (n_trials - failed_trials) / n_trials
        
        # Create result
        config = BenchmarkConfig(
            environment=environment,
            experiment_design=experiment_design,
            estimator=estimator,
            n_trials=n_trials,
            seed=seed
        )
        
        result = BenchmarkResult(
            config=config,
            true_ate=true_ate,
            estimates=estimates,
            execution_time=execution_time,
            success_rate=success_rate,
            metadata={
                'T': T,
                'failed_trials': failed_trials,
                'env_name': env.name,
                'design_name': design.name,
                'estimator_name': est.name
            }
        )
        
        self.logger.info(f"Completed benchmark in {execution_time:.2f}s, success rate: {success_rate:.2%}")
        
        # Save result
        self._save_result(result)
        
        return result
    
    def _run_individual_trials(self, env, design, est, n_trials: int, T: int, seed: int, **kwargs) -> Tuple[List[float], int]:
        """Run individual trials (fallback method)."""
        estimates = []
        failed_trials = 0
        
        for trial in range(n_trials):
            trial_seed = seed + trial * 1000
            
            try:
                # Reset components
                env.reset(trial_seed)
                design.reset(trial_seed)
                
                # Generate experimental data
                exp_kwargs = {k: v for k, v in kwargs.items() if k != 'seed'}
                rewards, actions, states = env.simulate_experiment(design, T=T, seed=trial_seed, **exp_kwargs)
                
                # Estimate treatment effect
                estimate = est.estimate(rewards, actions, states, **kwargs)
                estimates.append(estimate)
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {str(e)}")
                failed_trials += 1
        
        return estimates, failed_trials
    
    def run_suite(self, configs: List[BenchmarkConfig]) -> Dict[str, BenchmarkResult]:
        """Run multiple benchmark configurations."""
        results = {}
        
        for i, config in enumerate(configs):
            self.logger.info(f"Running configuration {i+1}/{len(configs)}")
            
            result = self.run(
                environment=config.environment,
                experiment_design=config.experiment_design,
                estimator=config.estimator,
                n_trials=config.n_trials,
                seed=config.seed,
                **config.estimator_params
            )
            
            config_name = f"{config.environment}_{config.experiment_design}_{config.estimator}"
            results[config_name] = result
        
        # Generate comparative report
        self._generate_suite_report(results)
        
        return results
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.environment}_{result.config.experiment_design}_{result.config.estimator}_{timestamp}.pkl"
        
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        
        # Also save summary as JSON
        summary_file = filepath.with_suffix('.json')
        with open(summary_file, 'w') as f:
            summary = result.summary_stats()
            summary['config'] = {
                'environment': result.config.environment,
                'experiment_design': result.config.experiment_design,
                'estimator': result.config.estimator,
                'n_trials': result.config.n_trials
            }
            json.dump(summary, f, indent=2)
    
    def _generate_suite_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comparative report for multiple results."""
        # Convert to DataFrame for analysis
        rows = []
        for name, result in results.items():
            stats = result.summary_stats()
            stats['config_name'] = name
            stats['environment'] = result.config.environment
            stats['design'] = result.config.experiment_design
            stats['estimator'] = result.config.estimator
            rows.append(stats)
        
        df = pd.DataFrame(rows)
        
        # Save detailed results
        df.to_csv(self.output_dir / "suite_results.csv", index=False)
        
        # Generate summary
        summary = df.groupby(['environment', 'design', 'estimator']).agg({
            'bias': 'mean',
            'abs_bias': 'mean', 
            'mse': 'mean',
            'success_rate': 'mean'
        }).round(4)
        
        summary.to_csv(self.output_dir / "suite_summary.csv")
        
        self.logger.info("Suite report generated")
    
    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components."""
        return {
            'environments': list(self.environments.keys()),
            'designs': list(self.designs.keys()),
            'estimators': list(self.estimators.keys())
        }


def create_benchmark_suite(env_names: List[str], 
                          design_names: List[str], 
                          estimator_names: List[str],
                          n_trials: int = 100) -> List[BenchmarkConfig]:
    """Create a comprehensive benchmark suite."""
    configs = []
    
    for env in env_names:
        for design in design_names:
            for estimator in estimator_names:
                config = BenchmarkConfig(
                    environment=env,
                    experiment_design=design,
                    estimator=estimator,
                    n_trials=n_trials
                )
                configs.append(config)
    
    return configs
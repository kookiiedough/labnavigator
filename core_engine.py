import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

# --- MVP Parameter Space Definition ---
# Simplified parameter space for CRISPR gene editing optimization
# Based on common variables and MVP scope
param_space = [
    # gRNA selection - Represented by an integer index for simplicity in MVP
    # In a real system, this would be more complex (sequence, predicted score, etc.)
    Categorical(["gRNA_A", "gRNA_B", "gRNA_C", "gRNA_D", "gRNA_E"], name="gRNA_Selection"),
    # Delivery method
    Categorical(["Lipofection", "Electroporation", "Viral Vector"], name="Delivery_Method"),
    # Cell line
    Categorical(["HEK293T", "HeLa", "MCF7"], name="Cell_Line"),
    # Reagent Concentration (e.g., Cas9 concentration in nM)
    Real(10, 1000, prior="log-uniform", name="Cas9_Concentration_nM"),
    # Guide RNA Concentration (nM)
    Real(10, 1000, prior="log-uniform", name="gRNA_Concentration_nM"),
    # Incubation Time (hours)
    Integer(24, 72, name="Incubation_Time_hr")
]

# --- Core Bayesian Optimization Engine Class ---

class LabNavigatorEngine:
    def __init__(self, parameter_space, n_initial_points=5, acq_func="EI", knowledge_base_path=None):
        """Initializes the Bayesian Optimization engine.

        Args:
            parameter_space (list): List of skopt dimension objects.
            n_initial_points (int): Number of random points to sample initially.
            acq_func (str): Acquisition function to use (e.g., "EI", "LCB", "PI").
            knowledge_base_path (str, optional): Path to CSV file with prior experimental data.
        """
        self.parameter_space = parameter_space
        self.param_names = [dim.name for dim in parameter_space]
        
        # Initialize optimizer with default settings
        self.optimizer = Optimizer(
            dimensions=parameter_space,
            random_state=42, # for reproducibility
            n_initial_points=n_initial_points,
            acq_func=acq_func # Expected Improvement is a common choice
        )
        
        self.history_x = [] # To store parameters suggested
        self.history_y = [] # To store corresponding results (e.g., editing efficiency)
        
        # Load knowledge base if provided
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, file_path):
        """Loads prior experimental data from a CSV file and updates the optimizer.
        
        Args:
            file_path (str): Path to the CSV file containing prior experimental data.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Verify that the columns match our parameter space
            expected_columns = self.param_names + ['Outcome_Efficiency']
            if not all(col in df.columns for col in expected_columns):
                missing = [col for col in expected_columns if col not in df.columns]
                print(f"Warning: Knowledge base is missing expected columns: {missing}")
                return False
            
            # Convert dataframe rows to parameter lists and outcomes
            X_prior = []
            y_prior = []
            
            for _, row in df.iterrows():
                # Extract parameters in the correct order
                params = [row[name] for name in self.param_names]
                outcome = row['Outcome_Efficiency']
                
                # Add to prior data lists
                X_prior.append(params)
                # Remember: optimizer minimizes, so negate outcome if higher is better
                y_prior.append(-outcome)
            
            # Update the optimizer with prior data
            if X_prior and y_prior:
                for x, y in zip(X_prior, y_prior):
                    self.optimizer.tell(x, y)
                    # Also update our history
                    self.history_x.append(x)
                    self.history_y.append(-y)  # Store original (positive) outcome
                
                print(f"Successfully loaded {len(X_prior)} prior experiments from knowledge base.")
                return True
            else:
                print("Warning: No valid data found in knowledge base.")
                return False
                
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False

    def suggest_next_experiment(self):
        """Suggests the next set of parameters to test.

        Returns:
            list: A list of parameter values for the next experiment.
        """
        suggested_params_list = self.optimizer.ask()
        # Store the suggestion before returning
        self.history_x.append(suggested_params_list)
        return suggested_params_list

    def record_experiment_result(self, params_list, outcome):
        """Records the outcome of an experiment and updates the optimizer's model.

        Args:
            params_list (list): The parameters used in the experiment (should match a previous suggestion).
            outcome (float): The measured outcome (e.g., editing efficiency). 
                             Note: skopt minimizes, so we might need to negate maximization objectives.
                             For MVP, let's assume higher outcome is better, so we'll use -outcome.
        """
        # Ensure the params_list matches the last suggestion if not explicitly provided
        # For simplicity in MVP, we assume the result corresponds to the last suggestion
        # In a real system, we'd need a more robust way to match results to suggestions.
        if not self.history_x or params_list != self.history_x[-1]:
             print("Warning: Recorded parameters do not match the last suggestion or history is empty.")
             # Optionally, find the matching parameters in history_x if needed

        # skopt minimizes the objective function. If higher outcome is better (e.g., efficiency),
        # we need to feed the negative of the outcome to the optimizer.
        objective_value = -float(outcome)
        self.optimizer.tell(params_list, objective_value)
        self.history_y.append(outcome) # Store the original outcome
        print(f"Recorded result: {outcome} for parameters: {params_list}")

    def get_best_result(self):
        """Returns the best parameters found so far and their outcome.

        Returns:
            tuple: (best_params_list, best_outcome) or (None, None) if no results recorded.
        """
        if not self.optimizer.Xi:
            return None, None

        best_index = np.argmin(self.optimizer.yi)
        best_params_list = self.optimizer.Xi[best_index]
        # Remember optimizer stores minimized objective (-outcome)
        best_objective_value = self.optimizer.yi[best_index]
        best_outcome = -best_objective_value # Convert back to original scale

        return best_params_list, best_outcome
    
    def get_parameter_importance(self):
        """Estimates the relative importance of each parameter based on available data.
        
        Returns:
            dict: Parameter names mapped to their estimated importance scores (0-100).
                  Returns empty dict if insufficient data.
        """
        # This is a simplified implementation for MVP
        # In a real system, this would use more sophisticated methods
        
        if len(self.optimizer.Xi) < 5:  # Need sufficient data
            return {}
        
        importance = {}
        param_names = self.get_parameter_names()
        
        # Simple correlation-based importance for continuous parameters
        # For categorical parameters, we'll use average outcome per category
        
        # Convert optimizer data to numpy arrays
        X = np.array(self.optimizer.Xi)
        y = -np.array(self.optimizer.yi)  # Convert back to original scale
        
        for i, name in enumerate(param_names):
            if isinstance(self.parameter_space[i], Categorical):
                # For categorical, calculate average outcome per category
                categories = self.parameter_space[i].categories
                category_scores = {}
                
                for cat in categories:
                    mask = X[:, i] == cat
                    if np.any(mask):
                        category_scores[cat] = np.mean(y[mask])
                
                if category_scores:
                    # Normalize to 0-100 scale
                    min_score = min(category_scores.values())
                    max_score = max(category_scores.values())
                    range_score = max_score - min_score
                    
                    if range_score > 0:
                        importance[name] = 100 * range_score / np.max(y)
                    else:
                        importance[name] = 0
            else:
                # For continuous, use absolute correlation
                if len(np.unique(X[:, i])) > 1:  # Need variation
                    corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    if not np.isnan(corr):
                        importance[name] = 100 * corr
                    else:
                        importance[name] = 0
                else:
                    importance[name] = 0
        
        return importance

    def get_parameter_names(self):
        """Returns the names of the parameters in the defined space."""
        return [dim.name for dim in self.parameter_space]

    def get_experiment_count(self):
        """Returns the number of experiments recorded so far."""
        return len(self.optimizer.yi)

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Initializing LabNavigator Engine...")
    
    # Test with knowledge base
    engine = LabNavigatorEngine(param_space, n_initial_points=3, 
                               knowledge_base_path="/home/ubuntu/knowledge_base.csv")
    
    param_names = engine.get_parameter_names()
    print(f"Parameter Space Names: {param_names}")
    print(f"Loaded {engine.get_experiment_count()} experiments from knowledge base")
    
    # Get best result from knowledge base
    best_params, best_result = engine.get_best_result()
    if best_params:
        print("\n--- Best Result from Knowledge Base ---")
        print(f"Best parameters found: {dict(zip(param_names, best_params))}")
        print(f"Best outcome: {best_result:.2f}")
    
    # Simulate running a few additional experiments
    num_experiments_to_simulate = 3
    print(f"\nSimulating {num_experiments_to_simulate} additional experiments...")
    
    for i in range(num_experiments_to_simulate):
        print(f"\n--- Experiment {i+1} ---")
        # 1. Get suggestion
        suggested_params = engine.suggest_next_experiment()
        print(f"Suggested Parameters: {dict(zip(param_names, suggested_params))}")

        # 2. Simulate running the experiment and getting a result
        #    (Replace this with actual experimental results in a real application)
        #    This dummy function returns higher values for 'Lipofection' and mid-range concentrations
        simulated_outcome = 0
        if suggested_params[1] == "Lipofection":
            simulated_outcome += 30
        if suggested_params[1] == "Electroporation":
             simulated_outcome += 10
        simulated_outcome += (1 - abs(suggested_params[3] - 500) / 490) * 40 # Peak near 500nM Cas9
        simulated_outcome += (1 - abs(suggested_params[4] - 400) / 490) * 20 # Peak near 400nM gRNA
        simulated_outcome += (suggested_params[5] / 72) * 10 # Longer incubation slightly better
        simulated_outcome = max(0, min(100, simulated_outcome + np.random.randn() * 5)) # Add noise

        print(f"Simulated Outcome: {simulated_outcome:.2f}")

        # 3. Record the result
        engine.record_experiment_result(suggested_params, simulated_outcome)

    # Get the best result after all experiments
    best_params, best_result = engine.get_best_result()
    if best_params:
        print("\n--- Optimization Complete ---")
        print(f"Best parameters found: {dict(zip(param_names, best_params))}")
        print(f"Best outcome: {best_result:.2f}")
        
        # Get parameter importance
        importance = engine.get_parameter_importance()
        if importance:
            print("\n--- Parameter Importance ---")
            for name, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"{name}: {score:.1f}%")
    else:
        print("\nNo experiments were run.")

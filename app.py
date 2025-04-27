import sys
sys.path.append("/home/ubuntu") # Add parent directory to path to import core_engine

import os
import io
import base64
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Import pandas for potential future use, though not strictly needed here
from flask import Flask, render_template, request, redirect, url_for

# Import the engine and parameter space from the core_engine module
from core_engine import LabNavigatorEngine, param_space

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for session management or flashing, good practice

# --- Initialize the LabNavigator Engine with Knowledge Base ---
KNOWLEDGE_BASE_PATH = "/home/ubuntu/knowledge_base.csv"

lab_engine = LabNavigatorEngine(param_space, 
                              n_initial_points=3, # Fewer initial random points needed if KB is rich
                              knowledge_base_path=KNOWLEDGE_BASE_PATH)
param_names = lab_engine.get_parameter_names()

# --- Initialize Experiment History from Engine --- 
# The engine now loads history from the KB, let's sync the dashboard history
experiment_history = []
if lab_engine.optimizer.Xi: # Check if the optimizer has data (from KB or otherwise)
    # Optimizer stores X (params) and y (negative outcomes)
    loaded_params = lab_engine.optimizer.Xi
    loaded_outcomes = -np.array(lab_engine.optimizer.yi) # Convert back to positive outcomes
    
    for i in range(len(loaded_params)):
        params_list = loaded_params[i]
        outcome = loaded_outcomes[i]
        params_dict = dict(zip(param_names, params_list))
        experiment_history.append({
            "params": params_dict,
            "params_list": params_list,
            "outcome": outcome
        })
    print(f"Initialized dashboard history with {len(experiment_history)} experiments from engine.")

# --- Plotting Function (Modified to handle potentially larger history) ---
def generate_progress_plot():
    """Generates a plot of outcomes over experiment iterations."""
    # Use the engine's history which includes KB + new experiments
    if not lab_engine.optimizer.Xi:
        return None
        
    outcomes = -np.array(lab_engine.optimizer.yi) # Get all outcomes (positive scale)
    iterations = list(range(1, len(outcomes) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iterations, outcomes, marker=".", linestyle="-", markersize=4)
    ax.set_title("Experiment Outcome vs. Iteration")
    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Outcome (e.g., Editing Efficiency)")
    ax.grid(True)

    # Find best outcome and highlight
    best_outcome_so_far = -np.inf
    best_outcomes_history = []
    for outcome in outcomes:
        if outcome > best_outcome_so_far:
            best_outcome_so_far = outcome
        best_outcomes_history.append(best_outcome_so_far)
    
    ax.plot(iterations, best_outcomes_history, marker="", linestyle="--", color="red", label="Best Outcome So Far")
    ax.legend()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
    plt.close(fig) # Close the figure to free memory
    return f"data:image/png;base64,{plot_url}"

# --- Flask Routes ---

@app.route("/")
def index():
    """Main dashboard page."""
    suggested_params_list = None
    suggested_params_dict = None
    last_suggestion_index = -1 # Relative to dashboard history
    current_suggestion_in_engine = None # The actual suggestion list from engine

    # Check if the engine has made a suggestion that hasn't been told yet
    # The engine's history_x stores suggestions made via .ask()
    # The optimizer's Xi stores parameters for which results were .tell()
    if len(lab_engine.history_x) > len(lab_engine.optimizer.Xi):
        current_suggestion_in_engine = lab_engine.history_x[-1]
        suggested_params_list = current_suggestion_in_engine
        suggested_params_dict = dict(zip(param_names, suggested_params_list))
        # Find the corresponding entry in dashboard history (should be the last one if added correctly)
        if experiment_history and experiment_history[-1]["outcome"] is None:
             last_suggestion_index = len(experiment_history) - 1
        else:
             # This case might happen if dashboard history is out of sync, try to recover
             print("Warning: Engine suggestion found, but dashboard history mismatch.")
             # We might need to add the suggestion to dashboard history here if it's missing
             if not any(h["params_list"] == suggested_params_list for h in experiment_history):
                 experiment_history.append({
                     "params": suggested_params_dict,
                     "params_list": suggested_params_list,
                     "outcome": None
                 })
                 last_suggestion_index = len(experiment_history) - 1

    best_params, best_result = lab_engine.get_best_result()
    best_params_dict = dict(zip(param_names, best_params)) if best_params else None

    plot_url = generate_progress_plot()
    param_importance = lab_engine.get_parameter_importance()

    # Ensure history displayed is up-to-date (including KB)
    # Rebuild history from engine data for consistency
    current_experiment_history = []
    if lab_engine.optimizer.Xi:
        loaded_params = lab_engine.optimizer.Xi
        loaded_outcomes = -np.array(lab_engine.optimizer.yi)
        for i in range(len(loaded_params)):
            params_list = loaded_params[i]
            outcome = loaded_outcomes[i]
            params_dict = dict(zip(param_names, params_list))
            current_experiment_history.append({
                "params": params_dict,
                "params_list": params_list,
                "outcome": outcome
            })
    # Add pending suggestion if any
    if suggested_params_dict and not any(h["params_list"] == suggested_params_list for h in current_experiment_history):
         current_experiment_history.append({
             "params": suggested_params_dict,
             "params_list": suggested_params_list,
             "outcome": None
         })


    return render_template("index.html",
                           param_names=param_names,
                           suggested_params=suggested_params_dict,
                           suggested_params_list=suggested_params_list,
                           last_suggestion_index=last_suggestion_index, # This index might need careful handling
                           experiment_history=current_experiment_history, # Use the freshly built history
                           best_params=best_params_dict,
                           best_result=best_result,
                           plot_url=plot_url,
                           param_importance=param_importance)

@app.route("/suggest", methods=["POST"])
def suggest():
    """Suggest the next experiment."""
    # Check if the engine is waiting for a result
    can_suggest = len(lab_engine.history_x) == len(lab_engine.optimizer.Xi)

    if can_suggest:
        suggested_params_list = lab_engine.suggest_next_experiment()
        suggested_params_dict = dict(zip(param_names, suggested_params_list))
        # Add to dashboard history with outcome as None initially
        # Check if it's already there from index route logic
        if not any(h["params_list"] == suggested_params_list for h in experiment_history):
            experiment_history.append({
                "params": suggested_params_dict,
                "params_list": suggested_params_list,
                "outcome": None
            })
    else:
        # Handle the case where suggestion is blocked (optional: add flash message)
        print("Suggestion blocked: Waiting for result of previous experiment.")
        pass

    return redirect(url_for("index"))

@app.route("/record", methods=["POST"])
def record():
    """Record the outcome of the last suggested experiment."""
    try:
        outcome_str = request.form["outcome"]
        # suggestion_index = int(request.form["suggestion_index"]) # Less reliable now
        params_list_str = request.form["params_list"] # Get the string representation

        # Basic validation
        if not outcome_str:
            raise ValueError("Outcome value cannot be empty.")
        
        outcome = float(outcome_str)
        # Example validation: efficiency 0-100 (adjust as needed)
        # if outcome < 0 or outcome > 100: 
        #      raise ValueError("Outcome must be between 0 and 100.")

        # Convert the string representation of the list back to a list
        import ast
        params_list = ast.literal_eval(params_list_str)

        # Check if this matches the last engine suggestion
        if lab_engine.history_x and params_list == lab_engine.history_x[-1] and len(lab_engine.history_x) > len(lab_engine.optimizer.Xi):
            lab_engine.record_experiment_result(params_list, outcome)
            # Update the corresponding entry in dashboard history
            for i in range(len(experiment_history) - 1, -1, -1):
                if experiment_history[i]["params_list"] == params_list and experiment_history[i]["outcome"] is None:
                    experiment_history[i]["outcome"] = outcome
                    break
        else:
            # Handle error: suggestion mismatch or already recorded
            print(f"Error: Recorded parameters {params_list} do not match the last pending suggestion {lab_engine.history_x[-1] if lab_engine.history_x else 'None'} or result already recorded.")
            pass 

    except ValueError as e:
        # Handle invalid input (optional: add flash message to show error on page)
        print(f"Invalid input: {e}")
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle other errors
        pass

    return redirect(url_for("index"))

if __name__ == "__main__":
    # Make sure the static directory exists
    if not os.path.exists("/home/ubuntu/dashboard/static"):
        os.makedirs("/home/ubuntu/dashboard/static")
    # Run on 0.0.0.0 to be accessible externally if needed (e.g., via deploy_expose_port)
    # Use debug=True cautiously as it reloads, potentially resetting in-memory state if not handled.
    # For this MVP, restarting the app manually after code changes is safer for state consistency.
    app.run(host="0.0.0.0", port=5000, debug=False) 


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

class F1VisualizationCallback(tf.keras.callbacks.Callback):
    """
    A callback to visualize F1 scores during training and across trials.
    
    This callback tracks F1 scores during training and creates visualizations
    to show the progression of F1 scores both within a trial and across trials.
    """
    
    def __init__(self, trial_id=None, output_dir='f1_visualizations'):
        super(F1VisualizationCallback, self).__init__()
        self.trial_id = trial_id
        self.output_dir = output_dir
        self.f1_scores = []
        self.epochs = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store best F1 scores for each trial
        # This is a class variable shared across instances
        if not hasattr(F1VisualizationCallback, 'trial_scores'):
            F1VisualizationCallback.trial_scores = {}
    
    def on_epoch_end(self, epoch, logs=None):
        """Track F1 scores at the end of each epoch."""
        logs = logs or {}
        val_f1 = logs.get('val_f1_score_metric')
        
        if val_f1 is not None:
            self.f1_scores.append(val_f1)
            self.epochs.append(epoch + 1)
            
            # Update the plot for this trial
            self._plot_trial_progress()
    
    def on_train_end(self, logs=None):
        """Save the best F1 score for this trial and update the trials comparison plot."""
        if self.f1_scores:
            best_f1 = max(self.f1_scores)
            
            # Store the best F1 score for this trial
            if self.trial_id is not None:
                F1VisualizationCallback.trial_scores[self.trial_id] = best_f1
                
                # Update the trials comparison plot
                self._plot_trials_comparison()
    
    def _plot_trial_progress(self):
        """Plot the F1 score progress for the current trial."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.f1_scores, 'bo-', linewidth=2, markersize=8)
        
        # Add title with trial information
        title = 'F1 Score During Training'
        if self.trial_id is not None:
            title += f' (Trial {self.trial_id})'
        plt.title(title)
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation F1 Score')
        plt.grid(True)
        
        # Add the best F1 score as a horizontal line
        best_f1 = max(self.f1_scores)
        plt.axhline(y=best_f1, color='r', linestyle='--', 
                    label=f'Best F1: {best_f1:.4f}')
        plt.legend()
        
        # Save the figure
        filename = 'f1_progress.png'
        if self.trial_id is not None:
            filename = f'f1_progress_trial_{self.trial_id}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def _plot_trials_comparison(self):
        """Plot a comparison of F1 scores across all trials."""
        if not F1VisualizationCallback.trial_scores:
            return
        
        # Sort trials by ID
        sorted_trials = sorted(F1VisualizationCallback.trial_scores.items())
        trial_ids = [trial_id for trial_id, _ in sorted_trials]
        f1_scores = [score for _, score in sorted_trials]
        
        # Find the best trial
        best_f1 = max(f1_scores)
        best_trial_idx = f1_scores.index(best_f1)
        best_trial_id = trial_ids[best_trial_idx]
        
        # Create the bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(trial_ids, f1_scores, color='skyblue')
        
        # Highlight the best trial
        bars[best_trial_idx].set_color('green')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.axhline(y=best_f1, color='r', linestyle='--', 
                    label=f'Best F1: {best_f1:.4f} (Trial {best_trial_id})')
        
        plt.xlabel('Trial')
        plt.ylabel('Best Validation F1 Score')
        plt.title('F1 Score Comparison Across Trials')
        plt.grid(axis='y')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'f1_trials_comparison.png'))
        plt.close()
        
        # Create the line plot to show progression
        plt.figure(figsize=(12, 6))
        plt.plot(trial_ids, f1_scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=best_f1, color='r', linestyle='--', 
                    label=f'Best F1: {best_f1:.4f} (Trial {best_trial_id})')
        
        # Add labels for each point
        for i, (_, f1) in enumerate(zip(trial_ids, f1_scores)):
            plt.annotate(f'{f1:.4f}', (trial_ids[i], f1), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
        
        plt.xlabel('Trial')
        plt.ylabel('Best Validation F1 Score')
        plt.title('F1 Score Progression During Hyperparameter Search')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'f1_progression.png'))
        plt.close()

# Function to create a callback for a specific trial
def create_f1_visualization_callback(trial_id):
    """Create an F1 visualization callback for a specific trial."""
    return F1VisualizationCallback(trial_id=trial_id)

# Function to visualize F1 scores from existing tuner results
def visualize_tuner_f1_scores(tuner, output_dir='f1_visualizations'):
    """Visualize F1 scores from existing tuner results."""
    import os
    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store trial data
    trials_data = {}
    
    # Get the trials directory
    tuner_dir = tuner.directory
    project_dir = os.path.join(tuner_dir, tuner.project_name)
    
    # Walk through the directory to find trial folders
    for root, dirs, files in os.walk(project_dir):
        if 'trial.json' in files:
            # Load the trial.json file
            with open(os.path.join(root, 'trial.json'), 'r') as f:
                trial_data = json.load(f)
            
            trial_id = os.path.basename(root)
            
            # Extract metrics if available
            if 'metrics' in trial_data and 'metrics' in trial_data['metrics']:
                metrics = trial_data['metrics']['metrics']
                
                # Look for val_f1_score_metric
                if 'val_f1_score_metric' in metrics:
                    f1_values = metrics['val_f1_score_metric']
                    
                    # Store the best F1 score for this trial
                    if f1_values:
                        best_f1 = max(f1_values)
                        trials_data[trial_id] = {
                            'best_f1': best_f1,
                            'f1_history': f1_values
                        }
    
    if not trials_data:
        print("No trial data found with F1 scores.")
        return
    
    # Sort trials by ID
    sorted_trials = sorted(trials_data.items())
    trial_ids = [trial_id for trial_id, _ in sorted_trials]
    best_f1_scores = [data['best_f1'] for _, data in sorted_trials]
    
    # Find the best trial
    best_f1 = max(best_f1_scores)
    best_trial_idx = best_f1_scores.index(best_f1)
    best_trial_id = trial_ids[best_trial_idx]
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(trial_ids, best_f1_scores, color='skyblue')
    
    # Highlight the best trial
    bars[best_trial_idx].set_color('green')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=best_f1, color='r', linestyle='--', 
                label=f'Best F1: {best_f1:.4f} (Trial {best_trial_id})')
    
    plt.xlabel('Trial')
    plt.ylabel('Best Validation F1 Score')
    plt.title('F1 Score Comparison Across Trials')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'f1_trials_comparison.png'))
    plt.show()
    
    # Create the line plot to show progression
    plt.figure(figsize=(12, 6))
    plt.plot(trial_ids, best_f1_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=best_f1, color='r', linestyle='--', 
                label=f'Best F1: {best_f1:.4f} (Trial {best_trial_id})')
    
    # Add labels for each point
    for i, (_, f1) in enumerate(zip(trial_ids, best_f1_scores)):
        plt.annotate(f'{f1:.4f}', (trial_ids[i], f1), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    plt.xlabel('Trial')
    plt.ylabel('Best Validation F1 Score')
    plt.title('F1 Score Progression During Hyperparameter Search')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'f1_progression.png'))
    plt.show()
    
    # Visualize F1 scores during training for the best trial
    plt.figure(figsize=(12, 6))
    f1_history = trials_data[best_trial_id]['f1_history']
    epochs = range(1, len(f1_history) + 1)
    plt.plot(epochs, f1_history, 'bo-', linewidth=2)
    plt.title(f'F1 Score During Training for Best Trial ({best_trial_id})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'f1_training_best_trial.png'))
    plt.show()
    
    return best_trial_id, best_f1 
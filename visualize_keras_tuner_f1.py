import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_tuner_results(directory='tuner_results', project_name='disaster_tweets'):
    """Load the results from a Keras Tuner search."""
    tuner_dir = os.path.join(directory, project_name)
    
    # Dictionary to store trial data
    trials_data = {}
    
    # Walk through the directory to find trial folders
    for root, dirs, files in os.walk(tuner_dir):
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
                            'f1_history': f1_values,
                            'hyperparameters': trial_data.get('hyperparameters', {})
                        }
    
    return trials_data

def visualize_f1_progression(trials_data, output_dir='.'):
    """Visualize the progression of F1 scores across trials."""
    if not trials_data:
        print("No trial data found with F1 scores.")
        return None, None
    
    # Sort trials by ID
    sorted_trials = sorted(trials_data.items(), key=lambda x: x[0])
    trial_ids = [trial_id for trial_id, _ in sorted_trials]
    best_f1_scores = [data['best_f1'] for _, data in sorted_trials]
    
    # Find the best trial
    best_f1 = max(best_f1_scores)
    best_trial_idx = best_f1_scores.index(best_f1)
    best_trial_id = trial_ids[best_trial_idx]
    
    # Create the line plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(trial_ids) + 1), best_f1_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=best_f1, color='r', linestyle='--', 
                label='Best F1: {:.4f} (Trial {})'.format(best_f1, best_trial_id))
    
    # Add labels for each point
    for i, (_, f1) in enumerate(zip(trial_ids, best_f1_scores)):
        plt.annotate('{:.4f}'.format(f1), (i + 1, f1), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Best Validation F1 Score')
    plt.title('F1 Score Progression During Hyperparameter Search')
    plt.xticks(range(1, len(trial_ids) + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_progression.png'))
    plt.close()
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, len(trial_ids) + 1), best_f1_scores, color='skyblue')
    
    # Highlight the best trial
    bars[best_trial_idx].set_color('green')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                '{:.4f}'.format(height), ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=best_f1, color='r', linestyle='--', 
                label='Best F1: {:.4f} (Trial {})'.format(best_f1, best_trial_id))
    
    plt.xlabel('Trial Number')
    plt.ylabel('Best Validation F1 Score')
    plt.title('F1 Score Comparison Across Trials')
    plt.xticks(range(1, len(trial_ids) + 1))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_score_comparison.png'))
    plt.close()
    
    return best_trial_id, best_f1

def visualize_f1_during_training(trials_data, trial_id=None, output_dir='.'):
    """Visualize F1 scores during training for a specific trial or all trials."""
    if not trials_data:
        print("No trial data found with F1 scores.")
        return
    
    plt.figure(figsize=(12, 6))
    
    if trial_id and trial_id in trials_data:
        # Visualize a specific trial
        f1_history = trials_data[trial_id]['f1_history']
        epochs = range(1, len(f1_history) + 1)
        plt.plot(epochs, f1_history, 'bo-', linewidth=2, label='Trial {}'.format(trial_id))
        plt.title('F1 Score During Training for Trial {}'.format(trial_id))
        output_file = os.path.join(output_dir, 'f1_score_training_trial_{}.png'.format(trial_id))
    else:
        # Visualize all trials
        for trial_id, data in trials_data.items():
            f1_history = data['f1_history']
            epochs = range(1, len(f1_history) + 1)
            plt.plot(epochs, f1_history, 'o-', linewidth=1, label='Trial {}'.format(trial_id))
        plt.title('F1 Score During Training for All Trials')
        output_file = os.path.join(output_dir, 'f1_score_training_all_trials.png')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize F1 scores from Keras Tuner results')
    parser.add_argument('--directory', type=str, default='tuner_results',
                        help='Directory containing the tuner results')
    parser.add_argument('--project', type=str, default='disaster_tweets',
                        help='Project name used in the tuner')
    parser.add_argument('--output', type=str, default='.',
                        help='Directory to save the visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the tuner results
    trials_data = load_tuner_results(args.directory, args.project)
    
    if not trials_data:
        print("No trial data found. Make sure the tuner_results directory exists and contains trial data.")
        return
    
    print("Found {} trials with F1 score data.".format(len(trials_data)))
    
    # Visualize F1 score progression across trials
    best_trial_id, best_f1 = visualize_f1_progression(trials_data, args.output)
    
    if best_trial_id is not None:
        print("Best trial: {} with F1 score: {:.4f}".format(best_trial_id, best_f1))
        
        # Visualize F1 scores during training for the best trial
        visualize_f1_during_training(trials_data, best_trial_id, args.output)
    
    # Visualize F1 scores during training for all trials
    visualize_f1_during_training(trials_data, output_dir=args.output)
    
    print("Visualizations saved to {}".format(args.output))

if __name__ == "__main__":
    main() 
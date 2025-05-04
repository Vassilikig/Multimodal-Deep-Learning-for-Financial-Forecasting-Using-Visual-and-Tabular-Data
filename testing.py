import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Add this line to make torch.load work with numpy arrays
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

def evaluate_test_set(model, test_dataset, device, output_dir, batch_size=40):
    """
    Evaluate directional prediction model on test set with comprehensive classification metrics.
    """
    model.eval()
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # Initialize containers for predictions and targets
    horizons = ['direction_1d', 'direction_5d', 'direction_10d']
    target_keys = ['return_1d', 'return_5d', 'return_10d']
    
    all_preds = {k: [] for k in horizons}
    all_targets = {k: [] for k in target_keys}
    
    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            images = batch['images'].to(device)
            tabular = batch['tabular'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            outputs = model(images, tabular)
            
            # Store predictions and ground truth
            for pred_key, target_key in zip(horizons, target_keys):
                all_preds[pred_key].append(outputs[pred_key].cpu().numpy())
                all_targets[target_key].append(targets[target_key].cpu().numpy())
    
    # Concatenate batches
    for pred_key, target_key in zip(horizons, target_keys):
        all_preds[pred_key] = np.concatenate(all_preds[pred_key])
        all_targets[target_key] = np.concatenate(all_targets[target_key])
    
    # Calculate metrics
    results = {}
    summary_table = []
    
    for pred_key, target_key in zip(horizons, target_keys):
        # Convert predicted probabilities to binary predictions
        pred_labels = (all_preds[pred_key] > 0.5).astype(np.int32)
        # Convert actual returns to binary labels
        target_labels = (all_targets[target_key] > 0).astype(np.int32)
        
        # Calculate classification metrics
        accuracy = accuracy_score(target_labels, pred_labels) * 100
        precision = precision_score(target_labels, pred_labels, zero_division=0) * 100
        recall = recall_score(target_labels, pred_labels, zero_division=0) * 100
        f1 = f1_score(target_labels, pred_labels, zero_division=0) * 100
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(target_labels, pred_labels, labels=[0, 1]).ravel()
        
        # Store results
        results[pred_key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_neg': tn,
            'false_pos': fp, 
            'false_neg': fn,
            'true_pos': tp
        }
        
        # Add to summary table
        summary_table.append([
            pred_key.replace('direction_', ''),
            f"{accuracy:.2f}%",
            f"{precision:.2f}%",
            f"{recall:.2f}%",
            f"{f1:.2f}%",
            f"{tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)" if (tp+fn) > 0 else "N/A",
            f"{tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)" if (tn+fp) > 0 else "N/A"
        ])
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(
        summary_table,
        columns=['Horizon', 'Accuracy', 'Precision', 'Recall', 'F1', 'Up Detection', 'Down Detection']
    )
    
    # Print summary
    print("\nDirectional Prediction Test Set Evaluation Summary:")
    print(summary_df.to_string(index=False))
    
    # Save metrics to CSV
    summary_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)
    print(f"Test metrics saved to {os.path.join(output_dir, 'test_metrics.csv')}")
    
    # Create a unified bar chart for easy comparison across horizons
    plt.figure(figsize=(12, 6))
    
    # Set up bar positions
    horizons_labels = [h.replace('direction_', '') for h in horizons]
    x = np.arange(len(horizons_labels))
    width = 0.2
    
    # Plot bars for each metric (all as percentages)
    plt.bar(x - width*1.5, [results[h]['accuracy'] for h in horizons], width, label='Accuracy')
    plt.bar(x - width/2, [results[h]['precision'] for h in horizons], width, label='Precision')
    plt.bar(x + width/2, [results[h]['recall'] for h in horizons], width, label='Recall')
    plt.bar(x + width*1.5, [results[h]['f1'] for h in horizons], width, label='F1 Score')
    
    # Add labels and title
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Percentage (%)')
    plt.title('Directional Prediction Performance Metrics')
    plt.xticks(x, horizons_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Metrics comparison plot saved to {fig_path}")
    
    # Create confusion matrices as separate figures
    plt.figure(figsize=(15, 5))
    for i, horizon in enumerate(horizons):
        plt.subplot(1, 3, i+1)
        
        # Extract values from results
        tn = results[horizon]['true_neg']
        fp = results[horizon]['false_pos']
        fn = results[horizon]['false_neg']
        tp = results[horizon]['true_pos']
        
        # Create confusion matrix for plotting
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {horizon.replace("direction_", "")}')
        plt.colorbar()
        
        classes = ['Down', 'Up']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrices saved to {cm_path}")
    
    return results

def evaluate_and_visualize_test_set():
    """
    Load the best model and evaluate on test set
    """
    # Create the configuration
    config = Config()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    print("Loading test dataset...")
    test_transform = StockDataset.get_transforms(is_training=False)
    
    test_dataset = StockDataset(
        csv_file=config.test_csv,
        base_dir=config.data_dir,
        image_transform=test_transform,
        max_seq_len=config.seq_length,
        standardize_targets=False  # Match your training setup
    )
    
    # Count tabular features to initialize model with correct dimensions
    tabular_input_dim = count_tabular_features(config.test_csv)
    
    # Create model with same architecture as training
    model = StockReturnPredictor(
        tabular_input_dim=tabular_input_dim,
        embedding_dim=config.image_embedding_dim
    )
    
    # Load best model weights
    best_model_path = os.path.join(config.output_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"Best model not found at {best_model_path}. Using latest checkpoint instead.")
        checkpoints = sorted([f for f in os.listdir(config.output_dir) if f.startswith('checkpoint_epoch')])
        if checkpoints:
            latest_checkpoint = os.path.join(config.output_dir, checkpoints[-1])
            print(f"Loading latest checkpoint from {latest_checkpoint}")
            # Set weights_only=False to fix the loading issue
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            raise FileNotFoundError("No model checkpoints found.")
    
    model = model.to(device)
    model.eval()
    
    # Create test results directory
    test_output_dir = os.path.join(config.output_dir, 'test_results')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Evaluate on test set
    results = evaluate_test_set(model, test_dataset, device, test_output_dir)
    
    return results

# Run the evaluation
if __name__ == "__main__":
    evaluate_and_visualize_test_set()
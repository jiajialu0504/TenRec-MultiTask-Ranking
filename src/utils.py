import torch
from sklearn.metrics import roc_auc_score

def evaluate_model(model, loader, device, target_features):
    model.eval()
    all_labels = {target: [] for target in target_features}
    all_preds = {target: [] for target in target_features}
    
    with torch.no_grad():
        for batch in loader:
            x_s, x_d = batch['x_sparse'].to(device), batch['x_dense'].to(device)
            labels = batch['labels']
            outputs = model(x_s, x_d)
            
            for target in target_features:
                all_labels[target].extend(labels[target].numpy())
                all_preds[target].extend(outputs[target].cpu().squeeze().numpy())
    
    auc_results = {}
    for target in target_features:
        try:
            auc_results[target] = roc_auc_score(all_labels[target], all_preds[target])
        except ValueError:
            auc_results[target] = 0.5 
    return auc_results
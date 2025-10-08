import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, f1_score, accuracy_score)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        print(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Average Precision (AP)
        avg_precision = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification Report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        self.evaluation_results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")
        if avg_precision:
            print(f"Average Precision: {avg_precision:.4f}")
        print()
        
        return self.evaluation_results[model_name]
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """Plot confusion matrix"""
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
    
    def plot_roc_curve(self, X_test, y_test, models_dict, figsize=(10, 8)):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=figsize)
        
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curve(self, X_test, y_test, models_dict, figsize=(10, 8)):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=figsize)
        
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                avg_precision = average_precision_score(y_test, y_pred_proba)
                
                plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.4f})')
        
        # Baseline (random classifier)
        baseline = y_test.sum() / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Random Classifier (AP = {baseline:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def create_model_comparison_table(self):
        """Create a comparison table of all evaluated models"""
        if not self.evaluation_results:
            print("No evaluation results available")
            return None
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Average Precision': results['average_precision'],
                'Precision (Class 1)': results['classification_report']['1']['precision'],
                'Recall (Class 1)': results['classification_report']['1']['recall'],
                'Support (Class 1)': results['classification_report']['1']['support']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        print("Model Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_feature_importance(self, model, feature_names, model_name="Model", top_n=20):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            sns.barplot(data=top_features, y='feature', x='importance')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print(f"{model_name} does not have feature_importances_ attribute")
            return None

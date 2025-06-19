#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Analysis Tool: Experimental Results Visualization and Performance Analysis Charts (with Save Function)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


def setup_english_font():
    """Setup English fonts"""
    try:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
    except:
        print("Warning: Font setup failed, using default fonts")


def calculate_param_counts(valid_results, input_size, hidden_size, num_layers, X_train):
    """Calculate model parameter counts"""
    param_counts = {
        'Basic LSTM': num_layers * (4 * (input_size + hidden_size) * hidden_size + 4 * hidden_size) + hidden_size,
        'LSTM(Hidden32)': num_layers * (4 * (input_size + 32) * 32 + 4 * 32) + 32,
        'LSTM(Hidden128)': num_layers * (4 * (input_size + 128) * 128 + 4 * 128) + 128,
        'LSTM+Attention': num_layers * (4 * (
                input_size + hidden_size) * hidden_size + 4 * hidden_size) + hidden_size ** 2 + hidden_size * 2,
        'MLP': X_train.shape[1] * 128 + 128 + 128 * 64 + 64 + 64 * 1 + 1
    }
    return param_counts


def create_save_directory():
    """Create save directory"""
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./stock_prediction_results_{timestamp}"

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"📁 Image save directory: {save_dir}")
    return save_dir


def save_figure(fig, filename, save_dir, dpi=300, formats=['png', 'pdf']):
    """Universal function for saving figures"""
    for fmt in formats:
        filepath = os.path.join(save_dir, f"{filename}.{fmt}")
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"   ✅ Saved: {filepath}")
        except Exception as e:
            print(f"   ❌ Save failed {filepath}: {e}")


def generate_comprehensive_plots(valid_results, y_test, std, mean, X_train, save_images=True):
    """Generate comprehensive visualization results (with save function)"""
    print("\n7. Generating visualization results...")

    # Create save directory
    save_dir = None
    if save_images:
        save_dir = create_save_directory()

    try:
        # Create main figure
        fig = plt.figure(figsize=(20, 15))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # 1. Prediction Performance Comparison
        plt.subplot(3, 3, 1)
        n_show = min(50, len(y_test))
        x_range = range(n_show)
        y_true_show = y_test[-n_show:] * std + mean

        plt.plot(x_range, y_true_show, 'k-', linewidth=3, label='True Values', alpha=0.8)

        for i, (model_name, result) in enumerate(valid_results.items()):
            pred_show = result['test_pred'][-n_show:]
            plt.plot(x_range, pred_show, '--', color=colors[i % len(colors)],
                     linewidth=2, label=model_name, alpha=0.8)

        plt.title('Stock Opening Price Prediction Performance', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps (Days)')
        plt.ylabel('Opening Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # 2. MAE Comparison Bar Chart
        plt.subplot(3, 3, 2)
        model_names = list(valid_results.keys())
        test_maes = [valid_results[name]['test_mae'] for name in model_names]

        bars = plt.bar(range(len(model_names)), test_maes,
                       color=colors[:len(model_names)], alpha=0.7, edgecolor='black')

        plt.title('Test MAE Comparison Across Models', fontsize=14, fontweight='bold')
        plt.ylabel('MAE')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

        # Add value labels on bars
        for bar, mae in zip(bars, test_maes):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(test_maes) * 0.01,
                     f'{mae:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')

        # 3. Training Loss Curves
        plt.subplot(3, 3, 3)
        for i, (model_name, result) in enumerate(valid_results.items()):
            if 'train_losses' in result and 'val_losses' in result:
                epochs_range = range(len(result['train_losses']))
                plt.plot(epochs_range, result['train_losses'], '-', color=colors[i % len(colors)],
                         label=f'{model_name} (Train)', alpha=0.7, linewidth=2)
                plt.plot(epochs_range, result['val_losses'], '--', color=colors[i % len(colors)],
                         label=f'{model_name} (Val)', alpha=0.7, linewidth=2)

        plt.title('Training Process Loss Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 4. Prediction Scatter Plot
        plt.subplot(3, 3, 4)
        for i, (model_name, result) in enumerate(valid_results.items()):
            y_true = y_test * std + mean
            y_pred = result['test_pred']
            plt.scatter(y_true, y_pred, alpha=0.6, s=30,
                        color=colors[i % len(colors)], label=model_name, edgecolors='black')

        # Perfect prediction line
        min_val = min(y_test * std + mean)
        max_val = max(y_test * std + mean)
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')

        plt.title('Predictions vs True Values Scatter Plot', fontsize=14, fontweight='bold')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Error Distribution Histogram
        plt.subplot(3, 3, 5)
        for i, (model_name, result) in enumerate(valid_results.items()):
            errors = result['test_pred'] - (y_test * std + mean)
            plt.hist(errors, bins=25, alpha=0.6, color=colors[i % len(colors)],
                     label=model_name, density=True, edgecolor='black')

        plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)

        # 6. Model Complexity Comparison
        plt.subplot(3, 3, 6)
        param_counts = calculate_param_counts(valid_results, 1, 64, 2, X_train)
        names = [name for name in valid_results.keys() if name in param_counts]
        params = [param_counts[name] for name in names]

        if names and params:
            bars = plt.bar(range(len(names)), params, color=colors[:len(names)], alpha=0.7, edgecolor='black')
            plt.title('Model Parameter Count Comparison', fontsize=14, fontweight='bold')
            plt.ylabel('Number of Parameters')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.yscale('log')

            # Add value labels on bars
            for bar, param in zip(bars, params):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                         f'{param:,}', ha='center', va='bottom', fontsize=9, rotation=90)

        # 7. Train vs Test Performance Comparison
        plt.subplot(3, 3, 7)
        train_maes = [valid_results[name]['train_mae'] for name in model_names]
        test_maes = [valid_results[name]['test_mae'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, train_maes, width, label='Train MAE', color='lightblue', alpha=0.7, edgecolor='black')
        plt.bar(x + width / 2, test_maes, width, label='Test MAE', color='lightcoral', alpha=0.7, edgecolor='black')

        plt.title('Training vs Testing Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('MAE')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # 8. Relative Performance Radar Chart
        plt.subplot(3, 3, 8)
        metrics = ['Accuracy', 'Stability', 'Efficiency', 'Generalization']

        # Calculate relative metrics (normalized to 0-1)
        mae_scores = [1 - (mae - min(test_maes)) / (max(test_maes) - min(test_maes)) for mae in test_maes]
        stability_scores = [
            1 - abs(valid_results[name]['test_mae'] - valid_results[name]['train_mae']) / valid_results[name][
                'train_mae']
            for name in model_names]

        if params:
            complexity_scores = [1 - (param_counts.get(name, 0) - min(params)) / (max(params) - min(params)) for name in
                                 model_names]
        else:
            complexity_scores = [0.5] * len(model_names)

        generalization_scores = [1 - (valid_results[name]['test_mae'] / valid_results[name]['train_mae'] - 1) for name
                                 in model_names]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the shape

        ax = plt.subplot(3, 3, 8, projection='polar')

        for i, name in enumerate(model_names):
            values = [mae_scores[i], stability_scores[i], complexity_scores[i], generalization_scores[i]]
            values += values[:1]  # Close the shape
            ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        plt.title('Comprehensive Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.3, 1.0))

        # 9. Prediction Trend Analysis
        plt.subplot(3, 3, 9)
        # Show long-term prediction trend of the best model
        best_model = min(valid_results.keys(), key=lambda x: valid_results[x]['test_mae'])
        best_result = valid_results[best_model]

        # Calculate moving averages
        window = 10
        true_ma = pd.Series(y_test * std + mean).rolling(window=window).mean()
        pred_ma = pd.Series(best_result['test_pred']).rolling(window=window).mean()

        plt.plot(true_ma, label=f'True Values (MA{window})', color='black', linewidth=2)
        plt.plot(pred_ma, label=f'{best_model} Pred (MA{window})', color='red', linewidth=2, linestyle='--')

        plt.title(f'Best Model ({best_model}) Trend Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Opening Price (Moving Average)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save main figure
        if save_images and save_dir:
            save_figure(fig, "01_comprehensive_analysis", save_dir)

        plt.show()

        # Generate and save individual high-quality charts
        if save_images and save_dir:
            generate_individual_plots(valid_results, y_test, std, mean, X_train, save_dir, colors)

        print("✅ Visualization charts generated successfully")
        if save_images:
            print(f"📁 All charts saved to: {save_dir}")

        return True

    except Exception as e:
        print(f"❌ Error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_individual_plots(valid_results, y_test, std, mean, X_train, save_dir, colors):
    """Generate individual high-quality charts"""
    print("🎨 Generating individual high-quality charts...")

    # 1. Prediction Performance Comparison (High Quality)
    fig, ax = plt.subplots(figsize=(12, 8))
    n_show = min(100, len(y_test))
    x_range = range(n_show)
    y_true_show = y_test[-n_show:] * std + mean

    ax.plot(x_range, y_true_show, 'k-', linewidth=3, label='True Values', alpha=0.9)

    for i, (model_name, result) in enumerate(valid_results.items()):
        pred_show = result['test_pred'][-n_show:]
        ax.plot(x_range, pred_show, '--', color=colors[i % len(colors)],
                linewidth=2.5, label=model_name, alpha=0.8)

    ax.set_title('Stock Opening Price Prediction Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Steps (Days)', fontsize=14)
    ax.set_ylabel('Opening Price', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "02_prediction_comparison", save_dir)
    plt.close()

    # 2. MAE Comparison (High Quality)
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(valid_results.keys())
    test_maes = [valid_results[name]['test_mae'] for name in model_names]

    bars = ax.bar(range(len(model_names)), test_maes,
                  color=colors[:len(model_names)], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_title('Test MAE Comparison Across Models', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)

    # Add value labels on bars
    for bar, mae in zip(bars, test_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(test_maes) * 0.01,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')

    save_figure(fig, "03_mae_comparison", save_dir)
    plt.close()

    # 3. Training Loss Curves (High Quality)
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (model_name, result) in enumerate(valid_results.items()):
        if 'train_losses' in result and 'val_losses' in result:
            epochs_range = range(len(result['train_losses']))
            ax.plot(epochs_range, result['train_losses'], '-', color=colors[i % len(colors)],
                    label=f'{model_name} (Train)', alpha=0.8, linewidth=2.5)
            ax.plot(epochs_range, result['val_losses'], '--', color=colors[i % len(colors)],
                    label=f'{model_name} (Validation)', alpha=0.8, linewidth=2.5)

    ax.set_title('Training Process Loss Curves', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    save_figure(fig, "04_training_curves", save_dir)
    plt.close()

    # 4. Prediction Scatter Plot (High Quality)
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, (model_name, result) in enumerate(valid_results.items()):
        y_true = y_test * std + mean
        y_pred = result['test_pred']
        ax.scatter(y_true, y_pred, alpha=0.7, s=40,
                   color=colors[i % len(colors)], label=model_name, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_test * std + mean)
    max_val = max(y_test * std + mean)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=3, label='Perfect Prediction')

    ax.set_title('Predictions vs True Values Scatter Plot', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "05_prediction_scatter", save_dir)
    plt.close()

    # 5. Performance Comparison Bar Chart (High Quality)
    fig, ax = plt.subplots(figsize=(12, 8))

    train_maes = [valid_results[name]['train_mae'] for name in model_names]
    test_maes = [valid_results[name]['test_mae'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, train_maes, width, label='Train MAE',
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width / 2, test_maes, width, label='Test MAE',
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_title('Training vs Testing Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(train_maes + test_maes) * 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    save_figure(fig, "06_performance_comparison", save_dir)
    plt.close()

    print("   ✅ All individual charts generated successfully")


def generate_summary_report(valid_results, save_dir=None):
    """Generate text summary report"""
    print("\n📋 Generating summary report...")

    # Create report content
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Stock Opening Price Prediction Experiment Summary Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of Models: {len(valid_results)}")
    report_lines.append("")

    # Performance ranking
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['test_mae'])
    report_lines.append("📊 Model Performance Ranking (by Test MAE ascending):")
    report_lines.append("-" * 50)
    for i, (model_name, result) in enumerate(sorted_models, 1):
        report_lines.append(f"{i:2d}. {model_name:<15} - Test MAE: {result['test_mae']:.6f}")
    report_lines.append("")

    # Detailed results
    report_lines.append("📈 Detailed Experimental Results:")
    report_lines.append("-" * 50)
    for model_name, result in valid_results.items():
        report_lines.append(f"Model: {model_name}")
        report_lines.append(f"  Train MAE: {result['train_mae']:.6f}")
        report_lines.append(f"  Test MAE: {result['test_mae']:.6f}")
        ratio = result['test_mae'] / result['train_mae']
        report_lines.append(f"  Generalization Ratio: {ratio:.3f}")
        if ratio < 1.1:
            status = "Excellent"
        elif ratio < 1.3:
            status = "Good"
        else:
            status = "Overfitting"
        report_lines.append(f"  Generalization: {status}")
        report_lines.append("")

    # Best model analysis
    best_model_name = sorted_models[0][0]
    best_result = sorted_models[0][1]
    report_lines.append("🏆 Best Model Analysis:")
    report_lines.append("-" * 50)
    report_lines.append(f"Best Model: {best_model_name}")
    report_lines.append(f"Best Test MAE: {best_result['test_mae']:.6f}")
    report_lines.append(f"Train MAE: {best_result['train_mae']:.6f}")
    report_lines.append(f"Generalization Ratio: {best_result['test_mae'] / best_result['train_mae']:.3f}")
    report_lines.append("")

    # Performance comparison
    worst_result = sorted_models[-1][1]
    improvement = worst_result['test_mae'] - best_result['test_mae']
    improvement_pct = (improvement / worst_result['test_mae']) * 100
    report_lines.append("📊 Performance Improvement Analysis:")
    report_lines.append("-" * 50)
    report_lines.append(f"Best model compared to worst model:")
    report_lines.append(f"  MAE Improvement: {improvement:.6f}")
    report_lines.append(f"  Improvement Percentage: {improvement_pct:.2f}%")
    report_lines.append("")

    # Technical features
    report_lines.append("⚙️ Technical Implementation Features:")
    report_lines.append("-" * 50)
    report_lines.append("• Fully implemented based on MatrixSlow framework")
    report_lines.append("• Self-implemented LSTM, Attention mechanism, MLP models")
    report_lines.append("• Complete forward and backward propagation")
    report_lines.append("• Implemented SGD and Adam optimizers")
    report_lines.append("• Support for batch training and model evaluation")
    report_lines.append("")

    report_lines.append("=" * 80)

    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    if save_dir:
        report_file = os.path.join(save_dir, "experiment_summary_report.txt")
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Summary report saved: {report_file}")
        except Exception as e:
            print(f"❌ Report save failed: {e}")

    return report_text


def print_performance_table(valid_results):
    """Print performance comparison table"""
    print(f"\n{'Model Name':<15} {'Train MAE':<12} {'Test MAE':<12} {'Generalization':<15} {'Parameters':<12}")
    print("-" * 80)

    # Calculate model parameter counts (simplified)
    param_counts = {
        'Basic LSTM': 50000,  # Estimated
        'LSTM(Hidden32)': 20000,
        'LSTM(Hidden128)': 150000,
        'LSTM+Attention': 60000,
        'MLP': 5000
    }

    for model_name, result in valid_results.items():
        generalization = "Excellent" if result['test_mae'] / result['train_mae'] < 1.1 else \
            "Good" if result['test_mae'] / result['train_mae'] < 1.3 else "Overfitting"
        params = param_counts.get(model_name, 0)
        print(
            f"{model_name:<15} {result['train_mae']:<12.4f} {result['test_mae']:<12.4f} {generalization:<15} {params:<12}")


def generate_analysis_report(valid_results):
    """Generate detailed analysis report"""
    print("\n" + "=" * 60)
    print("8. Detailed Analysis Report")
    print("=" * 60)

    print("\n【Model Performance Analysis】")
    best_model = min(valid_results.keys(), key=lambda x: valid_results[x]['test_mae'])
    worst_model = max(valid_results.keys(), key=lambda x: valid_results[x]['test_mae'])

    print(f"🏆 Best Model: {best_model} (Test MAE: {valid_results[best_model]['test_mae']:.4f})")
    print(f"📈 Performance Ranking:")
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['test_mae'])
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"   {i}. {name}: {result['test_mae']:.4f}")

    print("\n【LSTM Architecture Effect Analysis】")
    lstm_results = {k: v for k, v in valid_results.items() if 'LSTM' in k}
    if lstm_results:
        best_lstm = min(lstm_results.keys(), key=lambda x: lstm_results[x]['test_mae'])
        print(f"• Best LSTM Variant: {best_lstm}")

        # Analyze attention mechanism effect
        if 'LSTM+Attention' in valid_results and 'Basic LSTM' in valid_results:
            base_mae = valid_results['Basic LSTM']['test_mae']
            att_mae = valid_results['LSTM+Attention']['test_mae']
            improvement = base_mae - att_mae
            print(f"• Attention Mechanism Effect: {'+' if improvement > 0 else ''}{improvement:.4f} MAE change")
            print(
                f"  - {'Significant improvement' if improvement > 0.01 else 'Slight improvement' if improvement > 0 else 'Performance degradation'}")

    print("\n【Overfitting Analysis】")
    for model_name, result in valid_results.items():
        ratio = result['test_mae'] / result['train_mae']
        if ratio < 1.1:
            status = "✅ Excellent generalization"
        elif ratio < 1.3:
            status = "⚠️ Slight overfitting"
        else:
            status = "❌ Severe overfitting"
        print(f"• {model_name}: {status} (ratio: {ratio:.3f})")

    print("\n【Computational Efficiency Analysis】")
    print("• MLP: Fewest parameters, fastest computation, suitable for real-time prediction")
    print("• Basic LSTM: Medium complexity, strong temporal modeling capability")
    print("• LSTM+Attention: Highest complexity, strongest theoretical expressiveness")

    print("\n【Temporal Feature Importance Analysis】")
    if 'MLP' in valid_results and 'Basic LSTM' in valid_results:
        mlp_mae = valid_results['MLP']['test_mae']
        lstm_mae = valid_results['Basic LSTM']['test_mae']
        if lstm_mae < mlp_mae:
            print(f"• Temporal modeling important: LSTM improves {mlp_mae - lstm_mae:.4f} MAE over MLP")
            print("• Recommendation: Stock prices have significant temporal dependencies")
        else:
            print(f"• Limited temporal modeling value: MLP performance comparable or better")
            print("• Recommendation: Current dataset may have weak temporal dependencies")

    print("\n【Practical Deployment Recommendations】")
    print(f"🎯 Recommended deployment model: {best_model}")
    print(f"📊 Expected performance: MAE ≈ {valid_results[best_model]['test_mae']:.4f}")

    # Give specific recommendations based on best model
    if best_model == 'MLP':
        print("💡 Deployment strategy: Lightweight real-time prediction system")
        print("   - Advantages: Fast computation, low resource consumption")
        print("   - Applications: High-frequency trading, real-time response scenarios")
    elif 'LSTM' in best_model:
        print("💡 Deployment strategy: Temporal modeling prediction system")
        print("   - Advantages: Captures temporal dependencies, high prediction accuracy")
        print("   - Applications: Intraday price prediction, trend analysis")

    print(f"\n🔧 Technical Features Summary:")
    print("   ✅ Fully self-implemented based on MatrixSlow")
    print("   ✅ Standard LSTM units and attention mechanisms")
    print("   ✅ Complete automatic differentiation system")
    print("   ✅ Numerically stable training process")
    print("   ✅ Multi-model comparison analysis framework")

    return best_model
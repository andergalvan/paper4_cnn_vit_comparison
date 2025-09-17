import os
import pandas as pd
import numpy as np
from scipy import stats

# Path where results are stored
RESULTS_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/finetuning_evaluation/results'

# List of models
AVAILABLE_MODELS = ['inception_v3', 'resnet50', 'densenet201', 'mobilenet_v3_large', 'efficientnet_b0', 'vit_b_16']

# Methods and their corresponding file names
METHODS = {
    "openmax": "evaluation_metrics_openmax.csv",
    "softmax_thresholding": "evaluation_metrics_softmax_thresholding.csv"
}

summary = []  # Final results

for method, filename in METHODS.items():
    for model in AVAILABLE_MODELS:
        model_path = os.path.join(RESULTS_PATH, model)
        if not os.path.isdir(model_path):
            continue

        # Collect metrics for each run
        runs_data = []
        for run in os.listdir(model_path):
            run_path = os.path.join(model_path, run, filename)
            if os.path.exists(run_path):
                df = pd.read_csv(run_path)
                runs_data.append(df.iloc[0].to_dict())

        if not runs_data:
            continue

        # Convert runs into a DataFrame
        runs_df = pd.DataFrame(runs_data)

        # Compute statistics for each metric
        for metric in runs_df.columns:
            values = runs_df[metric].astype(float)
            mean = values.mean()
            std = values.std(ddof=1)
            n = len(values)
            ci95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
            summary.append({
                "Method": method,
                "Model": model,
                "Metric": metric,
                "Mean": mean,
                "Std": std,
                "CI95_lower": ci95[0],
                "CI95_upper": ci95[1]
            })

# Save results into a DataFrame
summary_df = pd.DataFrame(summary)

# Print results in a readable format
for method in METHODS.keys():
    print(f"\n==== Results for {method.upper()} ====")
    for model in AVAILABLE_MODELS:
        df_model = summary_df[(summary_df["Model"] == model) & (summary_df["Method"] == method)]
        if df_model.empty:
            continue

        print(f"\nModel: {model}")
        for _, row in df_model.iterrows():
            print(f"  Metric: {row['Metric']}, "
                  f"Mean: {row['Mean']:.2f}, Std: {row['Std']:.2f}, "
                  f"95% CI: [{row['CI95_lower']:.2f}, {row['CI95_upper']:.2f}]")

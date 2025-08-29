"""
analysis_complete.py
---------------------

This script performs comprehensive analysis tasks for the Checkpoint 01
assignment in Data Science and Machine Learning. It extends the work
completed on the Individual Household Electric Power Consumption dataset
by addressing the additional exercises (21–25) and tackles a second
dataset—Appliances Energy Prediction—to complete exercises 26–35.  The
final section replicates, using Python, the explorations normally done
in Orange Data Mining for exercises 36–40.

The script produces figures and CSV summaries in the `outputs` directory
for convenient inspection.  It is intended to be run from the repository
root after ensuring the required datasets are present:

* `hpc_data/household_power_consumption.txt` — the household dataset.
* `energydata_complete.csv` — the appliances energy prediction dataset.

Usage:
    python analysis_complete.py --hpc-file hpc_data/household_power_consumption.txt \
        --appliances-file energydata_complete.csv --out outputs

"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_household_data(path: str) -> pd.DataFrame:
    """Load household power consumption data and perform basic cleaning.

    Parameters
    ----------
    path : str
        Path to `household_power_consumption.txt`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with parsed datetime index and numeric columns.
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        dtype=str
    )
    # Combine Date and Time into a datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('DateTime', inplace=True)
    # Convert numeric columns
    num_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def task21_series_temporais_por_hora(df: pd.DataFrame, out_dir: str):
    """Create hourly time series of Global_active_power and identify peak hours.

    Saves a CSV with hourly averages and a bar chart highlighting peak hours.
    """
    # Resample to hourly frequency
    hourly = df['Global_active_power'].resample('H').mean()
    hourly.to_csv(os.path.join(out_dir, 'task21_hourly_mean.csv'), index_label='DateTime', header=['HourlyMean'])
    # Compute average across all days for each hour of the day
    hourly_by_hour = hourly.groupby(hourly.index.hour).mean()
    peak_hours = hourly_by_hour.sort_values(ascending=False).head(5)
    # Save peak hours to CSV
    peak_hours.to_csv(os.path.join(out_dir, 'task21_peak_hours.csv'), header=['AveragePower'])
    # Plot hourly averages across a day
    plt.figure()
    hourly_by_hour.plot(marker='o')
    plt.title('Média horária de Global_active_power ao longo do dia')
    plt.xlabel('Hora do dia')
    plt.ylabel('Potência ativa média (kW)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task21_hourly_profile.png'))
    plt.close()


def task22_autocorrelacao(df: pd.DataFrame, out_dir: str):
    """Compute autocorrelation at lags of 1h, 24h and 48h for Global_active_power."""
    # Use hourly resampled series
    hourly = df['Global_active_power'].resample('H').mean().dropna()
    acf1 = hourly.autocorr(lag=1)
    acf24 = hourly.autocorr(lag=24)
    acf48 = hourly.autocorr(lag=48)
    acf_df = pd.DataFrame({
        'Lag': ['1h', '24h', '48h'],
        'Autocorrelation': [acf1, acf24, acf48]
    })
    acf_df.to_csv(os.path.join(out_dir, 'task22_autocorrelations.csv'), index=False)
    return acf_df


def task23_pca(df: pd.DataFrame, out_dir: str):
    """Apply PCA on selected variables and return explained variance ratio."""
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    df_sel = df[features].dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_sel)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    # Save explained variance to CSV
    pd.DataFrame({
        'Component': ['PC1', 'PC2'],
        'ExplainedVariance': explained_variance
    }).to_csv(os.path.join(out_dir, 'task23_pca_explained_variance.csv'), index=False)
    # Return PCA components for further tasks
    pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'], index=df_sel.index)
    return pca_df, explained_variance


def task24_cluster_visualization(pca_df: pd.DataFrame, kmeans_labels: pd.Series, out_dir: str):
    """Generate a scatter plot of PCA components colored by K-means cluster labels."""
    plt.figure()
    for cluster in sorted(np.unique(kmeans_labels)):
        cluster_points = pca_df[kmeans_labels == cluster]
        plt.scatter(cluster_points['PC1'], cluster_points['PC2'], s=2, label=f'Cluster {cluster}')
    plt.title('Clusters no espaço das componentes principais (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task24_pca_clusters.png'))
    plt.close()


def task25_polynomial_regression(df: pd.DataFrame, out_dir: str):
    """Compare linear and polynomial regression (degree 2) predicting Global_active_power from Voltage."""
    df_clean = df[['Global_active_power', 'Voltage']].dropna()
    X = df_clean[['Voltage']]
    y = df_clean['Global_active_power']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Linear model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    lin_mae = mean_absolute_error(y_test, y_pred_lin)
    lin_r2 = r2_score(y_test, y_pred_lin)
    # Polynomial model (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train)
    y_pred_poly = poly_reg.predict(X_test_poly)
    poly_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    poly_mae = mean_absolute_error(y_test, y_pred_poly)
    poly_r2 = r2_score(y_test, y_pred_poly)
    # Save metrics
    metrics_df = pd.DataFrame({
        'Model': ['Linear', 'PolynomialDegree2'],
        'RMSE': [lin_rmse, poly_rmse],
        'MAE': [lin_mae, poly_mae],
        'R2': [lin_r2, poly_r2]
    })
    metrics_df.to_csv(os.path.join(out_dir, 'task25_regression_metrics.csv'), index=False)
    # Plot results (scatter + fitted curves)
    # Use a subset for plotting to avoid huge file sizes
    sample_df = df_clean.sample(n=5000, random_state=42)
    plt.figure()
    plt.scatter(sample_df['Voltage'], sample_df['Global_active_power'], s=2, alpha=0.5, label='Dados')
    # Create curve range
    voltage_range = np.linspace(sample_df['Voltage'].min(), sample_df['Voltage'].max(), 100).reshape(-1, 1)
    plt.plot(voltage_range, lin_reg.predict(voltage_range), label='Linear', linewidth=2)
    voltage_range_poly = poly.transform(voltage_range)
    plt.plot(voltage_range, poly_reg.predict(voltage_range_poly), label='Polinomial grau 2', linewidth=2)
    plt.title('Regressão de Global_active_power em função de Voltage')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Global_active_power (kW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task25_regression_comparison.png'))
    plt.close()


def load_appliances_data(path: str) -> pd.DataFrame:
    """Load Appliances Energy Prediction dataset."""
    df = pd.read_csv(path)
    return df


def task26_info_description(df: pd.DataFrame, out_dir: str):
    """Save info and descriptive statistics for appliances dataset."""
    info_path = os.path.join(out_dir, 'task26_info.txt')
    with open(info_path, 'w') as f:
        df.info(buf=f)
    df.describe().to_csv(os.path.join(out_dir, 'task26_describe.csv'))


def task27_distribution(df: pd.DataFrame, out_dir: str):
    """Create histogram and time series for Appliances variable."""
    # Histogram
    plt.figure()
    df['Appliances'].hist(bins=50)
    plt.title('Histograma de Appliances')
    plt.xlabel('Consumo de energia (Wh)')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task27_appliances_hist.png'))
    plt.close()
    # Time series
    # Create datetime index if available: dataset has date/time in column 'date'
    if 'date' in df.columns:
        ts = pd.to_datetime(df['date'])
        plt.figure()
        plt.plot(ts, df['Appliances'])
        plt.title('Série temporal de Appliances')
        plt.xlabel('Data')
        plt.ylabel('Consumo de energia (Wh)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'task27_appliances_timeseries.png'))
        plt.close()


def task28_correlations(df: pd.DataFrame, out_dir: str):
    """Compute correlations between Appliances and environmental variables (temperature and humidity)."""
    # Identify potential environmental columns: those starting with T or RH
    env_cols = [c for c in df.columns if c.startswith('T') or c.startswith('RH')]
    correlations = df[env_cols].corrwith(df['Appliances'])
    corr_df = correlations.sort_values(ascending=False).reset_index()
    corr_df.columns = ['Variable', 'Correlation']
    corr_df.to_csv(os.path.join(out_dir, 'task28_correlations.csv'), index=False)
    return corr_df


def task29_normalization(df: pd.DataFrame, out_dir: str):
    """Apply Min-Max scaling to numeric variables and save summary statistics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    df_scaled = pd.DataFrame(scaled, columns=numeric_cols)
    # Save summary of min and max after scaling
    summary = pd.DataFrame({
        'Variable': numeric_cols,
        'MinScaled': df_scaled.min(),
        'MaxScaled': df_scaled.max()
    })
    summary.to_csv(os.path.join(out_dir, 'task29_minmax_summary.csv'), index=False)
    return df_scaled


def task30_pca(df_scaled: pd.DataFrame, out_dir: str):
    """Perform PCA on scaled numeric variables and save explained variance."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    explained = pca.explained_variance_ratio_
    pd.DataFrame({
        'Component': ['PC1', 'PC2'],
        'ExplainedVariance': explained
    }).to_csv(os.path.join(out_dir, 'task30_pca_explained_variance.csv'), index=False)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    return pca_df, explained


def task31_linear_regression(df: pd.DataFrame, out_dir: str):
    """Train multiple linear regression to predict Appliances using all other numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('Appliances')
    X = df[numeric_cols]
    y = df['Appliances']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(os.path.join(out_dir, 'task31_linear_regression_metrics.csv'), index=False)
    return metrics_df


def task32_random_forest_regressor(df: pd.DataFrame, out_dir: str):
    """Train a Random Forest regressor to predict Appliances and compare RMSE with linear regression."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('Appliances')
    X = df[numeric_cols]
    y = df['Appliances']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2]
    })
    metrics_df.to_csv(os.path.join(out_dir, 'task32_random_forest_metrics.csv'), index=False)
    return metrics_df


def task33_kmeans(df_scaled: pd.DataFrame, out_dir: str):
    """Run K-means clustering with 3 to 5 clusters and save cluster centers and counts."""
    results = []
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_scaled)
        centers = km.cluster_centers_
        counts = pd.Series(labels).value_counts().sort_index()
        # Save centers and counts
        centers_df = pd.DataFrame(centers, columns=df_scaled.columns)
        centers_df.to_csv(os.path.join(out_dir, f'task33_kmeans_{k}_centers.csv'), index_label='Cluster')
        counts.to_csv(os.path.join(out_dir, f'task33_kmeans_{k}_counts.csv'), header=['Count'])
        results.append({'k': k, 'inertia': km.inertia_})
    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'task33_kmeans_inertia.csv'), index=False)


def task34_classification(df: pd.DataFrame, out_dir: str):
    """Create binary target and train logistic regression and random forest classifiers."""
    # Binary target: 1 if Appliances > median, else 0
    median = df['Appliances'].median()
    df['HighConsumption'] = (df['Appliances'] > median).astype(int)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('Appliances')
    numeric_cols.remove('HighConsumption')
    X = df[numeric_cols]
    y = df['HighConsumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    # Compute metrics
    metrics = []
    for name, y_pred in [('LogisticRegression', y_pred_log), ('RandomForest', y_pred_rf)]:
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual_Low', 'Actual_High'], columns=['Pred_Low', 'Pred_High'])
        cm_df.to_csv(os.path.join(out_dir, f'task35_confusion_matrix_{name}.csv'))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(out_dir, 'task35_classification_metrics.csv'), index=False)


def task36_sample_orange(df: pd.DataFrame, out_dir: str):
    """Select a 1% random sample and compare distribution of Global_active_power."""
    sample_df = df.sample(frac=0.01, random_state=42)
    # Compare distributions: compute summary stats
    stats_full = df['Global_active_power'].describe()
    stats_sample = sample_df['Global_active_power'].describe()
    summary = pd.DataFrame({
        'Full': stats_full,
        'Sample1%': stats_sample
    })
    summary.to_csv(os.path.join(out_dir, 'task36_sample_distribution_comparison.csv'))


def task37_distribution_orange(df: pd.DataFrame, out_dir: str):
    """Create histogram of Global_active_power (replicate Orange Distribution widget)."""
    plt.figure()
    df['Global_active_power'].hist(bins=50)
    plt.title('Distribuição de Global_active_power (Orange)')
    plt.xlabel('Global_active_power (kW)')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task37_distribution_global_active_power.png'))
    plt.close()


def task38_scatter_orange(df: pd.DataFrame, out_dir: str):
    """Generate scatter plot of Voltage vs Global_intensity and compute correlation."""
    corr = df['Voltage'].corr(df['Global_intensity'])
    pd.DataFrame({'Correlation': [corr]}).to_csv(os.path.join(out_dir, 'task38_correlation_voltage_intensity.csv'), index=False)
    plt.figure()
    # Use a sample for plotting to reduce file size
    sample = df[['Voltage', 'Global_intensity']].dropna().sample(n=5000, random_state=42)
    plt.scatter(sample['Voltage'], sample['Global_intensity'], s=2, alpha=0.5)
    plt.title('Dispersão Voltage vs Global_intensity')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Global_intensity (A)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task38_scatter_voltage_intensity.png'))
    plt.close()


def task39_kmeans_submeters(df: pd.DataFrame, out_dir: str):
    """Apply K-means clustering to sub-metering variables and visualize clusters."""
    df_sub = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_sub)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    centers = kmeans.cluster_centers_
    # Save centers and counts
    pd.DataFrame(centers, columns=['Sub1', 'Sub2', 'Sub3']).to_csv(os.path.join(out_dir, 'task39_kmeans_submeter_centers.csv'), index_label='Cluster')
    pd.Series(labels).value_counts().sort_index().to_csv(os.path.join(out_dir, 'task39_kmeans_submeter_counts.csv'), header=['Count'])
    # 2D scatter plot using first two sub-meters
    plt.figure()
    colors = ['blue', 'orange', 'green']
    for idx in range(3):
        points = data_scaled[labels == idx]
        plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.5, label=f'Cluster {idx}')
    plt.title('K-means em Sub_metering (clusters)')
    plt.xlabel('Sub_metering_1 (scaled)')
    plt.ylabel('Sub_metering_2 (scaled)')
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'task39_kmeans_submeters_scatter.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Complete analysis for Checkpoint 01.')
    parser.add_argument('--hpc-file', type=str, required=True, help='Path to household power consumption txt file')
    parser.add_argument('--appliances-file', type=str, required=True, help='Path to appliances energy prediction csv')
    parser.add_argument('--out', type=str, required=True, help='Directory to save outputs')
    args = parser.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    # Load household data
    df_hpc = load_household_data(args.hpc_file)
    # Tasks 21–25
    task21_series_temporais_por_hora(df_hpc, out_dir)
    task22_autocorrelacao(df_hpc, out_dir)
    pca_df, explained_variance_hpc = task23_pca(df_hpc, out_dir)
    # Use earlier K-means clusters from 3 clusters on daily aggregated features
    # Recompute cluster labels: cluster 3 for daily aggregated features
    daily_features = pd.DataFrame({
        'Active_power_mean': df_hpc['Global_active_power'].resample('D').mean(),
        'Reactive_power_mean': df_hpc['Global_reactive_power'].resample('D').mean(),
        'Voltage_mean': df_hpc['Voltage'].resample('D').mean(),
        'Intensity_mean': df_hpc['Global_intensity'].resample('D').mean(),
        'Total_Sub_metering': (df_hpc['Sub_metering_1'] + df_hpc['Sub_metering_2'] + df_hpc['Sub_metering_3']).resample('D').mean()
    }).dropna()
    scaler_daily = MinMaxScaler()
    daily_scaled = scaler_daily.fit_transform(daily_features)
    kmeans_daily = KMeans(n_clusters=3, random_state=42, n_init=10)
    daily_labels = kmeans_daily.fit_predict(daily_scaled)
    # Align daily_labels to hourly index for PCA plot: map daily label to each hour index
    label_series = pd.Series(daily_labels, index=daily_features.index)
    # Map to PCA index (which is at minute resolution) by date: use reindex with forward fill/backfill
    pca_labels = label_series.reindex(pca_df.index, method='ffill')
    task24_cluster_visualization(pca_df, pca_labels, out_dir)
    task25_polynomial_regression(df_hpc, out_dir)
    # Load appliances data
    df_appl = load_appliances_data(args.appliances_file)
    task26_info_description(df_appl, out_dir)
    task27_distribution(df_appl, out_dir)
    task28_correlations(df_appl, out_dir)
    df_appl_scaled = task29_normalization(df_appl, out_dir)
    pca_appl, explained_appl = task30_pca(df_appl_scaled, out_dir)
    task31_linear_regression(df_appl, out_dir)
    task32_random_forest_regressor(df_appl, out_dir)
    task33_kmeans(df_appl_scaled, out_dir)
    task34_classification(df_appl, out_dir)
    # Orange-style tasks using household dataset
    task36_sample_orange(df_hpc, out_dir)
    task37_distribution_orange(df_hpc, out_dir)
    task38_scatter_orange(df_hpc, out_dir)
    task39_kmeans_submeters(df_hpc, out_dir)

if __name__ == '__main__':
    main()
# medical_imaging/module3/radiomics.py
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, mannwhitneyu

def case_str_to_int(case_str: str) -> int:
    """
    Convert a case string to an integer ID.
    
    Example:
        "case_0" -> 0, "case_10" -> 10
    
    Parameters
    ----------
    case_str : str
        Case identifier string.
    
    Returns
    -------
    int
        Case ID as integer.
    """
    return int(case_str.replace("case_", ""))

def extract_roi(ct_file: str, mask_file: str) -> np.ndarray:
    """
    Extract the region of interest (ROI) from a CT scan using the ground truth mask.
    The ROI is defined by the smallest bounding box containing all nonzero voxels.
    
    Parameters
    ----------
    ct_file : str
        Path to the CT scan NIfTI file.
    mask_file : str
        Path to the segmentation mask NIfTI file.
    
    Returns
    -------
    np.ndarray
        3D ROI array, or None if extraction fails.
    """
    try:
        ct_data = nib.load(ct_file).get_fdata(dtype=np.float32)
        mask_data = nib.load(mask_file).get_fdata(dtype=np.float32)
    except Exception as e:
        print(f"Error loading {ct_file} or {mask_file}: {e}")
        return None
    
    coords = np.where(mask_data > 0)
    if len(coords[0]) == 0:
        print(f"Warning: Empty mask in {mask_file}")
        return None
    
    min_x, max_x = coords[0].min(), coords[0].max()
    min_y, max_y = coords[1].min(), coords[1].max()
    min_z, max_z = coords[2].min(), coords[2].max()
    
    roi = ct_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    return roi

def compute_energy(voxel_values: np.ndarray) -> float:
    """
    Compute Energy: sum(v_i^2) for all voxel intensities.
    
    Parameters
    ----------
    voxel_values : np.ndarray
        1D array of voxel intensities.
    
    Returns
    -------
    float
        Energy value.
    """
    return float(np.sum(voxel_values**2))

def compute_mad(voxel_values: np.ndarray) -> float:
    """
    Compute Mean Absolute Deviation (MAD): average absolute difference to mean.
    
    Parameters
    ----------
    voxel_values : np.ndarray
        1D array of voxel intensities.
    
    Returns
    -------
    float
        MAD value.
    """
    mean_val = np.mean(voxel_values)
    return float(np.mean(np.abs(voxel_values - mean_val)))

def compute_uniformity(voxel_values: np.ndarray, num_bins: int = 64, fixed_range: tuple = (-1000, 1000)) -> float:
    """
    Compute Uniformity based on a normalized histogram.
    
    Parameters
    ----------
    voxel_values : np.ndarray
        1D array of voxel intensities.
    num_bins : int, optional
        Number of histogram bins.
    fixed_range : tuple, optional
        Fixed intensity range (min, max) for histogram binning.
    
    Returns
    -------
    float
        Uniformity value.
    """
    min_intensity = np.min(voxel_values)
    max_intensity = np.max(voxel_values)
    if np.isclose(min_intensity, max_intensity):
        return 1.0
    bin_edges = np.linspace(fixed_range[0], fixed_range[1], num_bins+1)
    hist, _ = np.histogram(voxel_values, bins=bin_edges)
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum <= 0:
        return 0.0
    p = hist / hist_sum
    return float(np.sum(p**2))

def compute_radiomic_features(roi: np.ndarray, num_bins: int = 64, fixed_range: tuple = (-1000, 1000)) -> dict:
    """
    Compute radiomic features: Energy, MAD, and Uniformity from an ROI.
    
    Parameters
    ----------
    roi : np.ndarray
        3D ROI array.
    num_bins : int, optional
        Number of bins for histogram.
    fixed_range : tuple, optional
        Fixed intensity range for histogram.
    
    Returns
    -------
    dict
        Dictionary containing 'energy', 'mad', and 'uniformity'.
    """
    voxel_values = roi.flatten()
    energy = compute_energy(voxel_values)
    mad = compute_mad(voxel_values)
    uniformity = compute_uniformity(voxel_values, num_bins, fixed_range)
    return {
        "energy": energy,
        "mad": mad,
        "uniformity": uniformity
    }

def process_cases(data_path: str, labels_file: str, num_bins: int = 64, fixed_range: tuple = (-1000, 1000)) -> pd.DataFrame:
    """
    Process all cases: extract ROIs, compute radiomic features, and merge with diagnosis labels.
    
    Parameters
    ----------
    data_path : str
        Directory containing NIfTI files.
    labels_file : str
        CSV file path with diagnosis labels.
    num_bins : int, optional
        Number of bins for histogram.
    fixed_range : tuple, optional
        Fixed intensity range.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with features and labels.
    """
    labels_df = pd.read_csv(labels_file)
    print(f"CSV columns: {labels_df.columns.tolist()}")
    print(labels_df.head())
    label_dict = dict(zip(labels_df["ID"], labels_df["Diagnosis"]))
    
    case_ids = []
    for f in os.listdir(data_path):
        if f.endswith("_mask.nii"):
            case_str = f.replace("_mask.nii", "")
            ct_file = os.path.join(data_path, f"{case_str}.nii")
            if os.path.exists(ct_file):
                case_ids.append(case_str)
    print(f"Found {len(case_ids)} cases: {case_ids}")
    
    rows = []
    for case_str in case_ids:
        case_id = case_str_to_int(case_str)
        if case_id not in label_dict:
            print(f"Skipping {case_str}: No diagnosis label found in CSV")
            continue
        diagnosis = label_dict[case_id]
        ct_file = os.path.join(data_path, f"{case_str}.nii")
        mask_file = os.path.join(data_path, f"{case_str}_mask.nii")
        roi = extract_roi(ct_file, mask_file)
        if roi is None:
            print(f"Skipping {case_str}: Failed to extract ROI")
            continue
        features = compute_radiomic_features(roi, num_bins, fixed_range)
        row = {
            "case_str": case_str,
            "ID": case_id,
            "Diagnosis": diagnosis,
            **features
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    print("\nFeature DataFrame head:")
    print(df.head())
    df['binary_diagnosis'] = df['Diagnosis'].apply(lambda x: 1 if x == 1 else 0)
    print("\nAdded binary diagnosis (1=malignant; 0=benign):")
    print(df[['case_str', 'ID', 'Diagnosis', 'binary_diagnosis']].head())
    return df

def visualize_features(df: pd.DataFrame):
    """
    Create various plots to visualize the distribution and relationships of radiomic features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and diagnosis labels.
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    palette = sns.color_palette("Set2", df["Diagnosis"].nunique())
    
    # Pairplot
    g = sns.pairplot(
        df,
        vars=["energy", "mad", "uniformity"],
        hue="Diagnosis",
        palette=palette,
        corner=True,
        diag_kind="kde",
        plot_kws={"s": 40, "alpha": 0.8}
    )
    g.fig.suptitle("Radiomic Feature Pairwise Distribution", fontsize=16, y=1.03)
    g.fig.tight_layout()
    plt.show()
    
    # Boxplot
    melted = df.melt(
        id_vars=["Diagnosis"],
        value_vars=["energy", "mad", "uniformity"],
        var_name="Feature",
        value_name="Value"
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Feature", y="Value", hue="Diagnosis", data=melted, palette=palette)
    plt.title("Radiomic Feature Distribution by Diagnosis", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Energy vs. MAD
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="energy",
        y="mad",
        hue="Diagnosis",
        palette=palette,
        s=70,
        edgecolor="black",
        alpha=0.8
    )
    plt.title("Scatter Plot: Energy vs. MAD", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    # KDE plot: Uniformity
    plt.figure(figsize=(8, 6))
    for diag in sorted(df["Diagnosis"].unique()):
        subset = df[df["Diagnosis"] == diag]
        sns.kdeplot(
            subset["uniformity"],
            label=f"Diagnosis {diag}",
            linewidth=2,
            fill=True,
            alpha=0.4
        )
    plt.title("Density Distribution of Uniformity by Diagnosis", fontsize=15)
    plt.tight_layout()
    plt.show()

def classify_features(df: pd.DataFrame) -> dict:
    """
    Perform classification using the radiomic features with a Random Forest classifier.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and diagnosis labels.
    
    Returns
    -------
    dict
        Dictionary with accuracy, classification report, and feature importances.
    """
    X = df[["energy", "mad", "uniformity"]].values
    y = df["Diagnosis"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    feature_importances = dict(zip(["energy", "mad", "uniformity"], clf.feature_importances_))
    
    print(f"Random Forest Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    print("\nFeature Importances:")
    for feature, importance in feature_importances.items():
        print(f"  {feature}: {importance:.4f}")
    
    plt.figure(figsize=(8, 5))
    sorted_idx = clf.feature_importances_.argsort()
    plt.barh(
        [ ["energy", "mad", "uniformity"][i] for i in sorted_idx ],
        clf.feature_importances_[sorted_idx]
    )
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()
    
    return {
        "accuracy": accuracy,
        "report": report,
        "feature_importances": feature_importances
    }

def plot_feature_histograms(df: pd.DataFrame, num_bins: int = 20):
    """
    Plot histograms for each radiomic feature.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features.
    num_bins : int, optional
        Number of histogram bins.
    """
    features = ["energy", "mad", "uniformity"]
    titles = ["Energy", "Mean Absolute Deviation (MAD)", "Uniformity"]
    
    plt.figure(figsize=(18, 5))
    for i, (feature, title) in enumerate(zip(features, titles)):
        plt.subplot(1, 3, i+1)
        plt.hist(df[feature], bins=num_bins, color='steelblue', edgecolor='black', alpha=0.85)
        plt.title(f"{title} Histogram", fontsize=14)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_feature_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and print statistical properties of the radiomic features by diagnosis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and diagnosis labels.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame of grouped statistics.
    """
    stats = []
    for feature in ["energy", "mad", "uniformity"]:
        for diagnosis in df["Diagnosis"].unique():
            subset = df[df["Diagnosis"] == diagnosis]
            stats.append({
                "feature": feature,
                "diagnosis": diagnosis,
                "mean": subset[feature].mean(),
                "median": subset[feature].median(),
                "std": subset[feature].std(),
                "min": subset[feature].min(),
                "max": subset[feature].max(),
                "count": len(subset)
            })
    stats_df = pd.DataFrame(stats)
    print("Feature Statistics by Diagnosis Group:")
    for feature in ["energy", "mad", "uniformity"]:
        feature_stats = stats_df[stats_df["feature"] == feature]
        print(f"\n--- {feature.upper()} ---")
        for _, row in feature_stats.iterrows():
            diag_label = "Malignant" if row["diagnosis"] == 1 else "Benign"
            print(f"{diag_label} (n={row['count']}): Mean={row['mean']:.2e}, Median={row['median']:.2e}, Std={row['std']:.2e}, Range=[{row['min']:.2e}, {row['max']:.2e}]")
    return stats_df

def test_feature_significance(df: pd.DataFrame) -> dict:
    """
    Test the statistical significance of the radiomic features between malignant and benign groups.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and diagnosis labels.
    
    Returns
    -------
    dict
        Dictionary with t-test and Mann-Whitney U test results for each feature.
    """
    results = {}
    if 'binary_diagnosis' not in df.columns:
        df['binary_diagnosis'] = df['Diagnosis'].apply(lambda x: 1 if x == 1 else 0)
    
    benign = df[df["binary_diagnosis"] == 0]
    malignant = df[df["binary_diagnosis"] == 1]
    
    print(f"Sample sizes: Benign={len(benign)}, Malignant={len(malignant)}")
    print(f"{'Feature':<12} {'t-statistic':>12} {'t-test p-value':>15} {'Mann-Whitney U':>15} {'M-W p-value':>15} {'Significant?':>12}")
    for feature in ["energy", "mad", "uniformity"]:
        try:
            t_stat, p_t = ttest_ind(benign[feature], malignant[feature], equal_var=False)
            u_stat, p_mw = mannwhitneyu(benign[feature], malignant[feature])
            is_significant = (p_t < 0.05) or (p_mw < 0.05)
            sig_marker = "✓" if is_significant else "✗"
            print(f"{feature:<12} {t_stat:>12.4f} {p_t:>15.4f} {u_stat:>15.1f} {p_mw:>15.4f} {sig_marker:>12}")
            results[feature] = {
                "t_statistic": t_stat,
                "t_test_p_value": p_t,
                "mann_whitney_u": u_stat,
                "mann_whitney_p_value": p_mw,
                "significant": is_significant
            }
        except Exception as e:
            print(f"{feature:<12} Error: {str(e)}")
            results[feature] = {"error": str(e)}
    return results

def evaluate_features_with_cv(df: pd.DataFrame) -> dict:
    """
    Evaluate the radiomic features using cross-validation with several classifiers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and binary diagnosis.
    
    Returns
    -------
    dict
        Dictionary with cross-validation results.
    """
    X = df[["energy", "mad", "uniformity"]].values
    y = df["binary_diagnosis"].values
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
    }
    
    feature_sets = {
        "energy": [0],
        "mad": [1],
        "uniformity": [2],
        "energy+mad": [0, 1],
        "energy+uniformity": [0, 2],
        "mad+uniformity": [1, 2],
        "all_features": [0, 1, 2]
    }
    
    results = {}
    print("Cross-Validation Results:")
    for feature_name, indices in feature_sets.items():
        feature_X = X[:, indices]
        results[feature_name] = {}
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, feature_X, y, cv=cv, scoring='accuracy')
            results[feature_name][clf_name] = {
                "mean_accuracy": scores.mean(),
                "std_accuracy": scores.std(),
                "min_accuracy": scores.min(),
                "max_accuracy": scores.max()
            }
            print(f"{feature_name} - {clf_name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}, Min={scores.min():.4f}, Max={scores.max():.4f}")
    return results

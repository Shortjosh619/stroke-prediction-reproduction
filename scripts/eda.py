"""
Exploratory Data Analysis for Stroke Prediction Dataset

This script performs initial data exploration including:
- Dataset overview and summary statistics
- Missing value analysis
- Class distribution visualization
- Feature distribution analysis
- Correlation analysis

Author: Joshua Ajemiri
Date: February 2026
"""

from pathlib import Path
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

DATA_PATH = Path(r"C:\stroke-prediction-reproduction\data\raw\healthcare-dataset-stroke-data.csv")
RESULTS_DIR = Path(r"C:\stroke-prediction-reproduction\results")

def configure_plot_style():

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 10


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded data successfully from {filepath}")
    print(f"Data has {df.shape[0]} rows and {df.shape[1]} columns\n")
    return df 


def dataset_overview(df):
    """basic overview using pandas methods"""

    print("---DATASET OVERVIEW---")
    print(f"First 5 columns: \n{df.head()}")
    print(f"Summary Statistics: \n{df.describe()}")
    print(f"Missing values (pct): {df.isnull().sum()} ({df.isnull().sum() * 100})")


def analyse_target_distribution(df):
    "analyse and visualise the stroke variable distribution"

    stroke_counts = df["stroke"].value_counts()
    stroke_pct = df["stroke"].value_counts(normalize=True) * 100

    print("\nStroke Distribution:")
    print(f"No stroke: {stroke_counts[0]} ({stroke_pct[0]:.2f}%)")
    print(f"Stroke: {stroke_counts[1]} ({stroke_pct[1]:.2f}%)")
    print(f"Stroke imbalance Ratio: {stroke_counts[0]/stroke_counts[1]:.2f}:1")

    #Visuals
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    stroke_counts.plot(kind="bar", ax=axs[0], color=["royalblue", "lime"])
    axs[0].set_title("Stroke Distribution", fontsize=14, fontweight="bold")
    axs[0].set_xlabel("Stroke status")
    axs[0].set_ylabel("Count")
    axs[0].set_xticklabels(["No Stroke", "Stroke"])
    axs[0].grid(axis="y", alpha=0.3)

    axs[1].pie(stroke_counts, labels=["No stroke", "Stroke"], autopct="%1.1f%%", 
               colors=["red", "turquoise"], startangle=90)
    axs[1].set_title("Stroke Percentage", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR/ "class_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved as: class_distribution.png")
    plt.close()


def analyse_categorical_features(df):

    print("CATEGORICAL FEATURES ANALYSIS")

    categorical_cols= ["gender", "hypertension", "heart_disease", "ever_married", 
                        "work_type", "Residence_type", "smoking_status"]

    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3

    fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axs = axs.flatten()

    for idx, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        print(f"\n{col}:")
        print(value_counts)

        value_counts.plot(kind="bar", ax=axs[idx], color="darkgrey")
        axs[idx].set_title(f"{col} distribution", fontsize=12, fontweight="bold")
        axs[idx].set_xlabel("")
        axs[idx].set_ylabel("counts")
        axs[idx].tick_params(axis="x", rotation=45)
        axs[idx].grid(axis="y", alpha=0.3)

    for idx in range(len(categorical_cols), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "categorical_distributions.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved as: categorical_distributions.png")
    plt.close()


def analyse_numerical_features(df): 
    print("NUMERICAL FEATURE ANALYSIS")

    numerical_cols = ["age", "avg_glucose_level", "bmi"]
    n_cols = len(numerical_cols)

    fig, axs = plt.subplots(2, n_cols, figsize=(15, 10))
    axs = axs.flatten()

    for idx, col in enumerate(numerical_cols):
        axs[idx].hist(df[col].dropna(), bins=30, color="darkviolet", edgecolor="black", alpha=0.7)
        axs[idx].set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        axs[idx].set_xlabel(col)
        axs[idx].set_ylabel("Frequency")
        axs[idx].grid(axis="y", alpha=0.3)

    
        df.boxplot(column=col, by="stroke", ax=axs[idx + n_cols])
        axs[idx + n_cols].set_title(f"{col} by Stroke status", fontsize=12, fontweight="bold")
        axs[idx + n_cols].set_xlabel("Stroke (0=No 1=Yes)")
        axs[idx + n_cols].set_ylabel(col)

        print(f"\n{col} Stats:")
        print(df[col].describe())

    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "numerical_distributions.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved as: numerical_distributions.png")
    plt.close()


def correlations(df):
    print("\nCORRELATION ANALYSIS")
    numerical_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]

    corr_matrix = df[numerical_cols].corr()
    print("Correlation matrix:")
    print(corr_matrix)

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                 square=True, linewidths=1, cbar_kws={"shrink":0.7}, fmt=".3f")
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    print("\nSaved as: correlation_matrix.png")
    plt.close()

    #Stroke correlation
    print("\nCorrelations with stroke")
    stroke_corr = corr_matrix["stroke"].drop("stroke").abs().sort_values(ascending=False)
    print(stroke_corr)


def main():

    configure_plot_style()
    
    df = load_data(DATA_PATH)
    
    #Run all my analyses
    dataset_overview(df)
    analyse_target_distribution(df)
    analyse_categorical_features(df)
    analyse_numerical_features(df)
    correlations(df)


if __name__ == "__main__":
    main()
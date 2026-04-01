"""
Stage-based CICIDS2017 preprocessing pipeline.

Pipeline stages:
    1. Load & merge data
    2. Fix columns
    3. Clean data
    4. Labels
    5. Balance
    6. Feature engineering
    7. Scale
    8. Split
    9. Sequences
    10. Save outputs
    11. EDA report

Run the full pipeline through:
    from preprocessing_CICIDS2017 import Preprocess
    Preprocess().run()
"""

import os
import pickle
import warnings
import zipfile
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency
    SMOTE = None

warnings.filterwarnings("ignore")


class Preprocess:
    STAGES = (
        "1. Load & merge data",
        "2. Fix columns",
        "3. Clean data",
        "4. Labels",
        "5. Balance",
        "6. Feature engineering",
        "7. Scale",
        "8. Split",
        "9. Sequences",
        "10. Save outputs",
        "11. EDA report",
    )

    def __init__(
        self,
        dataset_dir: str = os.path.join("datasets", "MachineLearningCVE"),
        zip_path: str = os.path.join("datasets", "MachineLearningCSV.zip"),
        artifact_dir: str = "artifacts",
        output_dir: str = "preprocessed",
        output_filename: str = "training_dataset.csv",
        random_state: int = 42,
        test_size: float = 0.20,
        validation_size: float = 0.25,
        balance_strategy: str = "undersample",
        variance_threshold: float = 0.0,
        scaler_name: str = "standard",
        sequence_length: int = 10,
        sequence_stride: int = 1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.zip_path = zip_path
        self.artifact_dir = artifact_dir
        self.eda_dir = os.path.join(self.artifact_dir, "eda")
        self.eda_plot_dir = os.path.join(self.eda_dir, "plots")
        self.output_dir = output_dir
        self.split_output_dir = os.path.join(self.output_dir, "splits")
        self.sequence_output_dir = os.path.join(self.output_dir, "sequences")
        self.output_csv = os.path.join(output_dir, output_filename)
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        self.balance_strategy = balance_strategy.lower()
        self.variance_threshold = variance_threshold
        self.scaler_name = scaler_name.lower()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.benign_class_index = None

    def log_stage(self, title: str) -> None:
        print(f"\n{'=' * 18} {title} {'=' * 18}")

    def ensure_directories(self) -> None:
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(self.eda_dir, exist_ok=True)
        os.makedirs(self.eda_plot_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.split_output_dir, exist_ok=True)
        os.makedirs(self.sequence_output_dir, exist_ok=True)

    def resolve_dataset_dir(self) -> str:
        if os.path.isdir(self.dataset_dir):
            print(f"Using existing dataset directory: {self.dataset_dir}")
            return self.dataset_dir

        if os.path.isfile(self.zip_path):
            os.makedirs(self.dataset_dir, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, "r") as zip_file:
                zip_file.extractall(self.dataset_dir)
                print(f"Extracted {len(zip_file.namelist())} files to {self.dataset_dir}")
            return self.dataset_dir

        raise FileNotFoundError(
            f"Dataset not found. Expected CSVs in '{self.dataset_dir}' "
            f"or zip file at '{self.zip_path}'."
        )

    def find_csv_files(self, root_dir: str) -> List[str]:
        csv_files: List[str] = []
        for root, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, file_name))

        csv_files.sort()
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under '{root_dir}'.")

        print(f"Found {len(csv_files)} CSV files")
        for file_path in csv_files:
            print(f"  {file_path}")
        return csv_files

    def load_raw_file(self, file_path: str) -> pd.DataFrame:
        print(f"\nLoading {os.path.basename(file_path)}...")
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        print(f"  Raw shape: {df.shape}")
        return df

    def stage_load_and_merge_data(self) -> pd.DataFrame:
        self.log_stage(self.STAGES[0])
        dataset_dir = self.resolve_dataset_dir()
        csv_files = self.find_csv_files(dataset_dir)
        frames = [self.load_raw_file(file_path) for file_path in csv_files]
        merged_df = pd.concat(frames, ignore_index=True)
        print(f"\nMerged raw dataset shape: {merged_df.shape}")
        return merged_df

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        cleaned_df.columns = cleaned_df.columns.str.strip()
        return cleaned_df

    def stage_fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_stage(self.STAGES[1])
        fixed_df = self.clean_column_names(df)
        if "Label" not in fixed_df.columns:
            raise ValueError(f"Label column not found. Columns: {fixed_df.columns.tolist()}")

        fixed_df["Label"] = fixed_df["Label"].astype(str).str.strip()
        print(f"Columns normalized: {len(fixed_df.columns)}")
        return fixed_df

    def stage_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_stage(self.STAGES[2])
        feature_df = df.drop(columns=["Label"]).copy()

        for column in feature_df.columns:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

        clean_df = feature_df.copy()
        clean_df["Label"] = df["Label"]
        clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        before_drop = len(clean_df)
        clean_df.dropna(inplace=True)
        clean_df.drop_duplicates(inplace=True)
        clean_df.reset_index(drop=True, inplace=True)

        numeric_columns = clean_df.drop(columns=["Label"]).columns
        clean_df[numeric_columns] = clean_df[numeric_columns].astype(np.float32)

        removed_rows = before_drop - len(clean_df)
        print(f"Clean shape: {clean_df.shape} (removed {removed_rows:,} rows)")
        return clean_df

    def encode_labels(self, labels: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
        label_encoder = LabelEncoder()
        multiclass_labels = label_encoder.fit_transform(labels)
        classes = label_encoder.classes_.tolist()
        self.benign_class_index = classes.index("BENIGN") if "BENIGN" in classes else None
        if self.benign_class_index is None:
            binary_labels = np.ones(len(labels), dtype=np.int8)
        else:
            binary_labels = (multiclass_labels != self.benign_class_index).astype(np.int8)

        with open(os.path.join(self.artifact_dir, "label_encoder.pkl"), "wb") as file_obj:
            pickle.dump(label_encoder, file_obj)

        print("\nLabel mapping:")
        for index, label_name in enumerate(label_encoder.classes_):
            count = int((multiclass_labels == index).sum())
            print(f"  {index:2d} -> {label_name:<40s} count: {count:>8,}")

        return multiclass_labels.astype(np.int32), binary_labels, label_encoder

    def stage_encode_labels(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
        self.log_stage(self.STAGES[3])
        labels = df["Label"]
        multiclass_labels, binary_labels, label_encoder = self.encode_labels(labels)
        return df, multiclass_labels, binary_labels, label_encoder

    def select_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = df.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()
        if feature_df.empty:
            raise ValueError("No numeric features available after cleaning.")
        return feature_df

    def _label_counts(self, labels: np.ndarray) -> dict[int, int]:
        unique, counts = np.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}

    def _binary_from_multiclass(self, multiclass_labels: np.ndarray) -> np.ndarray:
        if self.benign_class_index is None:
            return np.ones(len(multiclass_labels), dtype=np.int8)
        return (multiclass_labels != self.benign_class_index).astype(np.int8)

    def balance_dataset(
        self,
        features: pd.DataFrame,
        multiclass_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        label_counts = self._label_counts(multiclass_labels)
        print(f"Original class counts: {label_counts}")

        if len(label_counts) <= 1:
            print("Skipping balancing because there is only one class.")
            return features.reset_index(drop=True), multiclass_labels, binary_labels

        if self.balance_strategy == "none":
            print("Skipping balancing because balance_strategy='none'.")
            return features.reset_index(drop=True), multiclass_labels, binary_labels

        if self.balance_strategy == "smote" and SMOTE is not None:
            sampler = SMOTE(random_state=self.random_state)
            balanced_x, balanced_mc = sampler.fit_resample(features, multiclass_labels)
            balanced_df = pd.DataFrame(balanced_x, columns=features.columns).astype(np.float32)
            balanced_mc = balanced_mc.astype(np.int32)
            balanced_bin = self._binary_from_multiclass(balanced_mc)
            print(f"Balanced class counts: {self._label_counts(balanced_mc)}")
            return balanced_df, balanced_mc, balanced_bin

        min_count = min(label_counts.values())
        rng = np.random.default_rng(self.random_state)
        selected_parts: List[np.ndarray] = []
        multiclass_labels = np.asarray(multiclass_labels)
        binary_labels = np.asarray(binary_labels)

        for class_id in sorted(label_counts):
            class_indices = np.flatnonzero(multiclass_labels == class_id)
            if len(class_indices) > min_count:
                class_indices = rng.choice(class_indices, size=min_count, replace=False)
            selected_parts.append(np.sort(class_indices))

        selected_indices = np.concatenate(selected_parts)
        rng.shuffle(selected_indices)

        balanced_df = features.iloc[selected_indices].reset_index(drop=True).astype(np.float32)
        balanced_mc = multiclass_labels[selected_indices].astype(np.int32)
        balanced_bin = binary_labels[selected_indices].astype(np.int8)
        print(f"Balanced class counts: {self._label_counts(balanced_mc)}")
        if self.balance_strategy == "smote" and SMOTE is None:
            print("SMOTE requested but imbalanced-learn is not installed; used undersampling instead.")
        return balanced_df, balanced_mc, balanced_bin

    def stage_balance_dataset(
        self,
        features: pd.DataFrame,
        multiclass_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        self.log_stage(self.STAGES[4])
        return self.balance_dataset(features, multiclass_labels, binary_labels)

    def stage_feature_engineering(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        self.log_stage(self.STAGES[5])
        variance_filter = VarianceThreshold(threshold=self.variance_threshold)
        filtered_array = variance_filter.fit_transform(feature_df)
        feature_names = feature_df.columns[variance_filter.get_support()].tolist()
        filtered_df = pd.DataFrame(filtered_array, columns=feature_names).astype(np.float32)

        feature_names_path = os.path.join(self.artifact_dir, "feature_names.txt")
        with open(feature_names_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("\n".join(feature_names))

        removed_features = feature_df.shape[1] - len(feature_names)
        print(f"Removed {removed_features} low-variance features")
        print(f"Final feature count: {len(feature_names)}")
        return filtered_df, feature_names

    def scale_features(self, features: pd.DataFrame) -> tuple[pd.DataFrame, object]:
        if self.scaler_name == "minmax":
            scaler = MinMaxScaler()
            scaler_filename = "scaler_minmax.pkl"
        else:
            scaler = StandardScaler()
            scaler_filename = "scaler_standard.pkl"

        scaled_array = scaler.fit_transform(features).astype(np.float32)
        scaled_df = pd.DataFrame(scaled_array, columns=features.columns)

        with open(os.path.join(self.artifact_dir, scaler_filename), "wb") as file_obj:
            pickle.dump(scaler, file_obj)

        return scaled_df, scaler

    def stage_scale_features(self, features: pd.DataFrame) -> tuple[pd.DataFrame, object]:
        self.log_stage(self.STAGES[6])
        return self.scale_features(features)

    def split_dataset(
        self,
        features: pd.DataFrame,
        multiclass_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        feature_array = features.to_numpy(dtype=np.float32)

        try:
            split_result = train_test_split(
                feature_array,
                multiclass_labels,
                binary_labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=multiclass_labels,
            )
        except ValueError:
            print("Falling back to non-stratified train/test split due to class sparsity.")
            split_result = train_test_split(
                feature_array,
                multiclass_labels,
                binary_labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        x_temp, x_test, y_mc_temp, y_mc_test, y_bin_temp, y_bin_test = split_result

        try:
            second_split = train_test_split(
                x_temp,
                y_mc_temp,
                y_bin_temp,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=y_mc_temp,
            )
        except ValueError:
            print("Falling back to non-stratified validation split due to class sparsity.")
            second_split = train_test_split(
                x_temp,
                y_mc_temp,
                y_bin_temp,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=None,
            )

        x_train, x_val, y_mc_train, y_mc_val, y_bin_train, y_bin_val = second_split

        print("\nDataset splits:")
        print(f"  Train:      {len(x_train):>10,}")
        print(f"  Validation: {len(x_val):>10,}")
        print(f"  Test:       {len(x_test):>10,}")

        return (
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
        )

    def stage_split_dataset(
        self,
        features: pd.DataFrame,
        multiclass_labels: np.ndarray,
        binary_labels: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        self.log_stage(self.STAGES[7])
        return self.split_dataset(features, multiclass_labels, binary_labels)

    def create_sequences(
        self,
        x_data: np.ndarray,
        y_multiclass: np.ndarray,
        y_binary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sequence_length <= 1:
            return x_data.astype(np.float32), y_multiclass.astype(np.int32), y_binary.astype(np.int8)

        if len(x_data) < self.sequence_length:
            return (
                np.empty((0, self.sequence_length, x_data.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int8),
            )

        sequence_list: List[np.ndarray] = []
        target_multiclass: List[int] = []
        target_binary: List[int] = []

        for start_idx in range(0, len(x_data) - self.sequence_length + 1, self.sequence_stride):
            end_idx = start_idx + self.sequence_length
            sequence_list.append(x_data[start_idx:end_idx])
            target_multiclass.append(int(y_multiclass[end_idx - 1]))
            target_binary.append(int(y_binary[end_idx - 1]))

        return (
            np.asarray(sequence_list, dtype=np.float32),
            np.asarray(target_multiclass, dtype=np.int32),
            np.asarray(target_binary, dtype=np.int8),
        )

    def stage_create_sequences(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_mc_train: np.ndarray,
        y_mc_val: np.ndarray,
        y_mc_test: np.ndarray,
        y_bin_train: np.ndarray,
        y_bin_val: np.ndarray,
        y_bin_test: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        self.log_stage(self.STAGES[8])
        sequence_payload = {
            "train": self.create_sequences(x_train, y_mc_train, y_bin_train),
            "validation": self.create_sequences(x_val, y_mc_val, y_bin_val),
            "test": self.create_sequences(x_test, y_mc_test, y_bin_test),
        }

        for split_name, (sequence_x, _, _) in sequence_payload.items():
            print(f"{split_name.title()} sequences: {sequence_x.shape}")

        return sequence_payload

    def build_split_frame(
        self,
        x_data: np.ndarray,
        y_multiclass: np.ndarray,
        y_binary: np.ndarray,
        feature_names: List[str],
        split_name: str,
    ) -> pd.DataFrame:
        split_df = pd.DataFrame(x_data, columns=feature_names)
        split_df["label_multiclass"] = y_multiclass.astype(np.int32)
        split_df["label_binary"] = y_binary.astype(np.int8)
        split_df["split"] = split_name
        return split_df

    def save_training_dataset(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_mc_train: np.ndarray,
        y_mc_val: np.ndarray,
        y_mc_test: np.ndarray,
        y_bin_train: np.ndarray,
        y_bin_val: np.ndarray,
        y_bin_test: np.ndarray,
        feature_names: List[str],
    ) -> pd.DataFrame:
        final_df = pd.concat(
            [
                self.build_split_frame(x_train, y_mc_train, y_bin_train, feature_names, "train"),
                self.build_split_frame(x_val, y_mc_val, y_bin_val, feature_names, "validation"),
                self.build_split_frame(x_test, y_mc_test, y_bin_test, feature_names, "test"),
            ],
            ignore_index=True,
        )

        final_df.to_csv(self.output_csv, index=False)
        print(f"\nSaved training-ready dataset to {self.output_csv}")
        print(f"Final dataset shape: {final_df.shape}")
        return final_df

    def save_split_outputs(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_mc_train: np.ndarray,
        y_mc_val: np.ndarray,
        y_mc_test: np.ndarray,
        y_bin_train: np.ndarray,
        y_bin_val: np.ndarray,
        y_bin_test: np.ndarray,
        feature_names: List[str],
        sequence_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        split_payload = {
            "train": (x_train, y_mc_train, y_bin_train),
            "validation": (x_val, y_mc_val, y_bin_val),
            "test": (x_test, y_mc_test, y_bin_test),
        }

        for split_name, (x_data, y_mc, y_bin) in split_payload.items():
            x_df = pd.DataFrame(x_data, columns=feature_names)
            y_df = pd.DataFrame(
                {
                    "label_multiclass": y_mc.astype(np.int32),
                    "label_binary": y_bin.astype(np.int8),
                }
            )
            x_df.to_csv(os.path.join(self.split_output_dir, f"X_{split_name}.csv"), index=False)
            y_df.to_csv(os.path.join(self.split_output_dir, f"y_{split_name}.csv"), index=False)
            np.save(os.path.join(self.split_output_dir, f"X_{split_name}.npy"), x_data.astype(np.float32))
            np.save(os.path.join(self.split_output_dir, f"y_{split_name}_multiclass.npy"), y_mc.astype(np.int32))
            np.save(os.path.join(self.split_output_dir, f"y_{split_name}_binary.npy"), y_bin.astype(np.int8))

        for split_name, (sequence_x, sequence_y_mc, sequence_y_bin) in sequence_payload.items():
            np.save(os.path.join(self.sequence_output_dir, f"sequences_{split_name}.npy"), sequence_x)
            np.save(
                os.path.join(self.sequence_output_dir, f"sequence_labels_{split_name}_multiclass.npy"),
                sequence_y_mc,
            )
            np.save(
                os.path.join(self.sequence_output_dir, f"sequence_labels_{split_name}_binary.npy"),
                sequence_y_bin,
            )

    def stage_save_outputs(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        y_mc_train: np.ndarray,
        y_mc_val: np.ndarray,
        y_mc_test: np.ndarray,
        y_bin_train: np.ndarray,
        y_bin_val: np.ndarray,
        y_bin_test: np.ndarray,
        feature_names: List[str],
        sequence_payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        self.log_stage(self.STAGES[9])
        final_df = self.save_training_dataset(
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
            feature_names,
        )
        self.save_split_outputs(
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
            feature_names,
            sequence_payload,
        )
        return final_df

    def summarize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()
        return numeric_df.describe().transpose()

    def save_eda_reports(
        self,
        dataset_shape: tuple[int, int],
        duplicate_count: int,
        null_counts: pd.Series,
        label_distribution: pd.Series,
        numeric_summary: pd.DataFrame,
    ) -> None:
        null_counts_df = null_counts.rename("null_count").reset_index()
        null_counts_df.columns = ["column", "null_count"]
        label_distribution_df = label_distribution.rename("count").reset_index()
        label_distribution_df.columns = ["label", "count"]

        null_counts_df.to_csv(os.path.join(self.eda_dir, "null_counts.csv"), index=False)
        label_distribution_df.to_csv(os.path.join(self.eda_dir, "label_distribution.csv"), index=False)
        numeric_summary.to_csv(os.path.join(self.eda_dir, "numeric_summary.csv"))

        report_path = os.path.join(self.eda_dir, "eda_report.txt")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("CICIDS2017 EDA Report\n")
            file_obj.write("====================\n\n")
            file_obj.write(f"Dataset shape: {dataset_shape}\n")
            file_obj.write(f"Duplicate rows: {duplicate_count}\n")
            file_obj.write(f"Total null values: {int(null_counts.sum())}\n")
            file_obj.write(f"Columns with nulls: {int((null_counts > 0).sum())}\n\n")

            file_obj.write("Label distribution:\n")
            for label_name, count in label_distribution.items():
                file_obj.write(f"  {label_name}: {int(count)}\n")

            if not numeric_summary.empty:
                file_obj.write("\nTop numeric features by variance:\n")
                variances = numeric_summary.get("std", pd.Series(dtype=float)).sort_values(ascending=False)
                for feature_name, std_value in variances.head(10).items():
                    file_obj.write(f"  {feature_name}: std={float(std_value):.4f}\n")

    def save_eda_plots(
        self,
        null_counts: pd.Series,
        label_distribution: pd.Series,
        numeric_df: pd.DataFrame,
    ) -> None:
        if plt is None:
            print("Skipping EDA plots because matplotlib is not installed.")
            return

        plt.style.use("ggplot")

        label_plot_path = os.path.join(self.eda_plot_dir, "label_distribution.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        label_distribution.plot(kind="bar", ax=ax, color="#c44e52")
        ax.set_title("Label Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(label_plot_path, dpi=200)
        plt.close(fig)

        missing_plot_path = os.path.join(self.eda_plot_dir, "missing_values.png")
        missing_to_plot = null_counts[null_counts > 0].sort_values(ascending=False).head(20)
        if missing_to_plot.empty:
            missing_to_plot = null_counts.sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_to_plot.plot(kind="bar", ax=ax, color="#4c72b0")
        ax.set_title("Top Columns by Missing Values")
        ax.set_xlabel("Column")
        ax.set_ylabel("Null Count")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(missing_plot_path, dpi=200)
        plt.close(fig)

        if numeric_df.empty:
            return

        sampled_numeric = numeric_df
        if len(sampled_numeric) > 5000:
            sampled_numeric = sampled_numeric.sample(n=5000, random_state=self.random_state)

        variance_series = sampled_numeric.var().sort_values(ascending=False).head(15)
        variance_plot_path = os.path.join(self.eda_plot_dir, "top_feature_variance.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        variance_series.plot(kind="bar", ax=ax, color="#55a868")
        ax.set_title("Top Numeric Features by Variance")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Variance")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(variance_plot_path, dpi=200)
        plt.close(fig)

        corr_source = sampled_numeric.loc[:, variance_series.index]
        if corr_source.shape[1] >= 2:
            correlation = corr_source.corr()
            heatmap_path = os.path.join(self.eda_plot_dir, "correlation_heatmap.png")
            fig, ax = plt.subplots(figsize=(12, 10))
            image = ax.imshow(correlation, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            ax.set_title("Correlation Heatmap")
            ax.set_xticks(range(len(correlation.columns)))
            ax.set_xticklabels(correlation.columns, rotation=90)
            ax.set_yticks(range(len(correlation.index)))
            ax.set_yticklabels(correlation.index)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=200)
            plt.close(fig)

    def stage_eda_report(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_stage(self.STAGES[10])
        null_counts = df.isna().sum().sort_values(ascending=False)
        duplicate_count = int(df.duplicated().sum())
        label_distribution = df["Label"].value_counts().sort_values(ascending=False)
        numeric_summary = self.summarize_numeric_features(df)
        numeric_df = df.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()

        print(f"Dataset shape: {df.shape}")
        print(f"Duplicate rows: {duplicate_count}")
        print(f"Total null values: {int(null_counts.sum())}")
        print(f"Columns with nulls: {int((null_counts > 0).sum())}")
        print("\nLabel distribution:")
        for label_name, count in label_distribution.items():
            print(f"  {label_name:<30s} {int(count):>10,}")

        if not numeric_summary.empty:
            print("\nNumeric summary preview:")
            print(numeric_summary[["mean", "std", "min", "max"]].head(10).round(4).to_string())

        self.save_eda_reports(df.shape, duplicate_count, null_counts, label_distribution, numeric_summary)
        self.save_eda_plots(null_counts, label_distribution, numeric_df)
        print(f"\nEDA artifacts saved to {self.eda_dir}")
        return df

    def run(self) -> pd.DataFrame:
        self.ensure_directories()
        raw_df = self.stage_load_and_merge_data()
        fixed_df = self.stage_fix_columns(raw_df)
        clean_df = self.stage_clean_data(fixed_df)
        encoded_df, multiclass_labels, binary_labels, _ = self.stage_encode_labels(clean_df)
        numeric_features = self.select_numeric_features(encoded_df)
        balanced_features, balanced_mc, balanced_bin = self.stage_balance_dataset(
            numeric_features,
            multiclass_labels,
            binary_labels,
        )
        engineered_features, feature_names = self.stage_feature_engineering(balanced_features)
        scaled_features, _ = self.stage_scale_features(engineered_features)

        (
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
        ) = self.stage_split_dataset(scaled_features, balanced_mc, balanced_bin)

        sequence_payload = self.stage_create_sequences(
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
        )

        final_df = self.stage_save_outputs(
            x_train,
            x_val,
            x_test,
            y_mc_train,
            y_mc_val,
            y_mc_test,
            y_bin_train,
            y_bin_val,
            y_bin_test,
            feature_names,
            sequence_payload,
        )

        self.stage_eda_report(clean_df)

        print("\nPreprocessing complete")
        print(f"Rows: {len(final_df):,}")
        print(f"Features: {len(feature_names):,}")
        print("Output files:")
        print(f"  {self.output_csv}")
        print(f"  {os.path.join(self.split_output_dir, 'X_train.csv')}")
        print(f"  {os.path.join(self.split_output_dir, 'y_train.csv')}")
        print(f"  {os.path.join(self.sequence_output_dir, 'sequences_train.npy')}")
        print(f"  {os.path.join(self.artifact_dir, 'label_encoder.pkl')}")
        if self.scaler_name == "minmax":
            print(f"  {os.path.join(self.artifact_dir, 'scaler_minmax.pkl')}")
        else:
            print(f"  {os.path.join(self.artifact_dir, 'scaler_standard.pkl')}")
        print(f"  {os.path.join(self.artifact_dir, 'feature_names.txt')}")
        print(f"  {os.path.join(self.eda_dir, 'eda_report.txt')}")
        return final_df


if __name__ == "__main__":
    Preprocess().run()

__all__ = ["Preprocess"]

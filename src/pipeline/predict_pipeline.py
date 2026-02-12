import pickle
import pandas as pd

def load_artifacts(model_path, preprocessor_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def predict_Target(input_dict, model_path, preprocessor_path):
    model, preprocessor = load_artifacts(model_path, preprocessor_path)

    # Expected raw columns
    expected_cols = preprocessor.feature_names_in_

    # Create input DF with ALL expected columns
    input_df = pd.DataFrame(columns=expected_cols)

    # Fill provided values
    for col, val in input_dict.items():
        if col in expected_cols:
            input_df.loc[0, col] = val

    # Auto-fill missing columns
    for col in expected_cols:
        if col not in input_df or pd.isna(input_df.loc[0, col]):
            input_df.loc[0, col] = 0  # safe default (scaler-friendly)

    transformed_data = preprocessor.transform(input_df)
    prediction = model.predict(transformed_data)[0]

    return prediction

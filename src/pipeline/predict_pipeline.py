import pickle
import pandas as pd

def load_artifacts(model_path, preprocessor_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def apply_business_defaults(input_df, preprocessor):

    defaults = {

        # Behaviour Risk
        "OBS_30_CNT_SOCIAL_CIRCLE": 3,
        "OBS_60_CNT_SOCIAL_CIRCLE": 3,
        "DEF_30_CNT_SOCIAL_CIRCLE": 1,
        "DEF_60_CNT_SOCIAL_CIRCLE": 1,

        # Location Risk
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,

        # Stability Risk
        "DAYS_LAST_PHONE_CHANGE": 200,

        # Document Risk
        "FLAG_DOCUMENT_3": 0,
        "FLAG_EMP_PHONE": 0,

        # Employment Risk
        "EMPLOYED_YEARS": 2,

        # Family Burden
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 2
    }

    # -------- Fill Defaults --------

    for col in input_df.columns:
        if input_df[col].isnull().any():
            if col in defaults:
                input_df[col] = defaults[col]
            else:
                input_df[col] = 0

    # -------- FIX DTYPE (CRITICAL) --------

    train_dtypes = preprocessor.feature_names_in_

    for col in input_df.columns:
        try:
            input_df[col] = input_df[col].astype(float)
        except:
            pass

    return input_df



def predict_Target(input_dict, model_path, preprocessor_path):

    model, preprocessor = load_artifacts(model_path, preprocessor_path)

    expected_cols = preprocessor.feature_names_in_

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=expected_cols)

    # APPLY DEFAULTS + DTYPE FIX
    input_df = apply_business_defaults(input_df, preprocessor)

    transformed_data = preprocessor.transform(input_df)

    transformed_df = pd.DataFrame(
        transformed_data,
        columns=preprocessor.get_feature_names_out()
    )

    prob = model.predict_proba(transformed_df)[:,1]

    return prob[0]



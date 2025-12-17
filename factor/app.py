# app.py

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ======================================================
# LOAD TRAINED MODEL + FEATURES
# ======================================================
model = joblib.load("behavior_impact_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================================================
# HELPER: Encode input behaviors
# ======================================================
def encode_behaviors(behaviors):
    input_vector = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    for behavior, value in behaviors.items():
        value = str(value).strip()

        # Binary YES / NO
        if value.lower() in ["yes", "no"]:
            if value.lower() == "yes" and behavior in input_vector.columns:
                input_vector.at[0, behavior] = 1

        # Categorical
        else:
            col_name = f"{behavior}_{value}"
            if col_name in input_vector.columns:
                input_vector.at[0, col_name] = 1

    return input_vector

# ======================================================
# API ENDPOINT
# ======================================================
@app.route("/impact-behavior", methods=["POST"])
def impact_behavior():

    data = request.json

    inr_status = data.get("inr_status")
    behaviors = data.get("behaviors")

    if inr_status is None or behaviors is None:
        return jsonify({"error": "inr_status and behaviors required"}), 400

    X = encode_behaviors(behaviors)

    # Predict probability
    prob = model.predict_proba(X)[0][1]

    # Feature impact (local importance)
    impact_scores = X.iloc[0] * model.feature_importances_

    impact_df = (
        pd.DataFrame({
            "feature": impact_scores.index,
            "impact_score": impact_scores.values
        })
        .sort_values("impact_score", ascending=False)
    )

    top_impacts = []
    for _, row in impact_df.head(5).iterrows():
        if row["impact_score"] > 0:
            top_impacts.append({
                "behavior": row["feature"],
                "impact_score": round(float(row["impact_score"]), 4)
            })

    return jsonify({
        "inr_status": "Out of Range" if inr_status == 1 else "Stable",
        "model_probability_out_of_range": round(float(prob), 3),
        "top_impact_behaviors": top_impacts
    })

# ======================================================
if __name__ == "__main__":
    app.run(debug=True, port = 8090)

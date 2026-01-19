import gradio as gr
import pandas as pd
import joblib
import numpy as np
from transformers import pipeline

# -------------------------------------------------
# Load models once at startup (not on every request)
# -------------------------------------------------

# BERT models
distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
roberta = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")



# -------------------------------------------------
# Core prediction logic
# -------------------------------------------------

def predict_one(model_name: str, text: str):
    """
    Predict sentiment for a single text with the chosen model.
    Returns a dict: {"label": <str>, "confidence": <float>}
    """
    text = text or ""

    if model_name == "distilbert":
        print("Using distilbert model for prediction...")
        outputs = distilbert([text], truncation=True)
        label = outputs[0]["label"]
        confidence = float(outputs[0]["score"])
        return {"label": label.lower(), "confidence": confidence}

    elif model_name == "roberta":
        print("Using roberta model for prediction...")
        outputs = roberta([text], truncation=True)
        label = outputs[0]["label"]
        confidence = float(outputs[0]["score"])
        return {"label": label.lower(), "confidence": confidence}

    elif model_name == "transformer_finetuned":
        print("Using transformer fine-tuned model for prediction...")
        # TODO : implement loading and prediction for fine-tuned model

    else:
        raise ValueError(f"Unknown model name: {model_name}")


def predict_one_gradio(model_name: str, text: str) -> str:
    """
    Wrapper for Gradio: returns a nicely formatted string
    instead of a dict.
    """
    if not text:
        return "Please enter some text."

    result = predict_one(model_name, text)
    return f"{result['label']} (confidence: {result['confidence']:.3f})"


def predict_csv(model_name: str, file_obj):
    """
    Expects a CSV with a column named 'review' (case-insensitive fallback: 'text').
    Returns: (preview_df, output_csv_path) for Gradio.

    The selected model_name is used for all rows.
    """
    if file_obj is None:
        return None, None

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                file_obj.name,
                encoding=enc,
                sep=";",
                engine="python",
                # on_bad_lines="skip",  
            )
        except Exception as e:
            print(f"Failed to read CSV with encoding {enc}: {e}")

    
    # Find the review/text column robustly
    cols_lower = {c.lower(): c for c in df.columns}
    if "review" in cols_lower:
        review_col = cols_lower["review"]
    elif "text" in cols_lower:
        review_col = cols_lower["text"]
    else:
        raise gr.Error("CSV must contain a column named 'review' (or 'text').")

    reviews = df[review_col].astype(str).fillna("").tolist()

    labels = []
    confidences = []
    pos_scores = []

    if model_name == "distilbert":
        outputs = distilbert(reviews, truncation=True)
        for out in outputs:
            label = out["label"]
            conf = float(out["score"])

            # Convert to P(positive) for convenience
            if label.upper().startswith("POS"):
                p_pos = conf
            else:
                p_pos = 1.0 - conf

            labels.append(label)
            confidences.append(conf)
            pos_scores.append(p_pos)

    elif model_name == "roberta":
        outputs = roberta(reviews, truncation=True)
        for out in outputs:
            label = out["label"]
            conf = float(out["score"])

            # Convert to P(positive) for convenience
            if label.upper().startswith("POS"):
                p_pos = conf
            else:
                p_pos = 1.0 - conf

            labels.append(label)
            confidences.append(conf)
            pos_scores.append(p_pos)

    else:
        raise gr.Error(f"Unknown model name: {model_name}")

    out_df = df.copy()
    out_df["pred_label"] = labels
    out_df["confidence"] = confidences
    # out_df["pos_score"] = pos_scores

    out_path = f"sentiment_predictions_{model_name}.csv"
    out_df.to_csv(out_path, index=False)

    # Show a small preview in the UI + allow download
    preview = out_df.head(25)
    return preview, out_path


# -------------------------------------------------
# Gradio UI
# -------------------------------------------------

with gr.Blocks(title="Sentiment Analysis Demo") as demo:
    gr.Markdown("# Sentiment Analysis Demo")
    gr.Markdown(
        "Choose a single review prediction or a batch CSV prediction "
        "(expects a `review` column)."
    )

    # Global model selector (used in both tabs)
    model_selector = gr.Radio(
        choices=["distilbert", "roberta"],
        value="svm",
        label="Choose model"
    )

    with gr.Tab("Single review"):
        inp = gr.Textbox(lines=6, label="Review text")
        out = gr.Textbox(label="Prediction")
        btn = gr.Button("Predict")
        btn.click(
            predict_one_gradio,
            inputs=[model_selector, inp],
            outputs=out
        )

    with gr.Tab("Batch CSV"):
        gr.Markdown(
            "Upload a CSV with a column named **review** "
            "(or **text**)."
        )
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        preview_out = gr.Dataframe(label="Preview (first 25 rows)")
        file_out = gr.File(label="Download predictions CSV")
        btn2 = gr.Button("Run batch prediction")
        btn2.click(
            predict_csv,
            inputs=[model_selector, file_in],
            outputs=[preview_out, file_out]
        )

if __name__ == "__main__":
    demo.launch()

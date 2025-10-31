import streamlit as st
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------------------------------------------------------------
# 1️⃣ Load Local Free Model (Flan-T5)
# ---------------------------------------------------------------------
@st.cache_resource
def get_llm():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )
    return generator


# ---------------------------------------------------------------------
# 2️⃣ Safe CSV Loader
# ---------------------------------------------------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                df.columns = df.columns.str.strip().str.lower()
                return df
        except Exception:
            continue

    raise ValueError(f"Could not read CSV: {path}")


# ---------------------------------------------------------------------
# 3️⃣ Load & Merge Property Data
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        project = safe_read_csv("project.csv")
        address = safe_read_csv("ProjectAddress.csv")
        config = safe_read_csv("ProjectConfiguration.csv")
        variant = safe_read_csv("ProjectConfigurationVariant.csv")

        for df in [project, address, config, variant]:
            df.columns = df.columns.str.strip().str.lower()

        project.rename(columns={"id": "project_id"}, inplace=True)
        address.rename(columns={"id": "address_id"}, inplace=True)
        config.rename(columns={"id": "config_id"}, inplace=True)
        variant.rename(columns={"id": "variant_id"}, inplace=True)

        merged = (
            project.merge(address, how="left", left_on="project_id", right_on="projectid")
                   .merge(config, how="left", left_on="project_id", right_on="projectid")
                   .merge(variant, how="left", left_on="config_id", right_on="configurationid")
        )

        merged = merged.loc[:, ~merged.columns.duplicated()].copy()

        if merged.empty:
            raise ValueError("Merged data is empty")

        return merged

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------
# 4️⃣ Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Property Search", page_icon="🏠", layout="wide")
st.title("🏠 AI Property Search")
st.write("Ask me anything about our property listings...")

data = load_data()

if data.empty:
    st.warning("⚠️ No data loaded. Ensure all CSV files are in the same folder as main.py.")
else:
    st.success(f"✅ Data loaded successfully! {len(data)} rows available.")
    st.dataframe(data.head())

    user_query = st.text_input("💬 Ask about properties (e.g., 'Show me 2BHK in Pune under 50L'):").strip()

    if user_query:
        llm = get_llm()

        with st.spinner("🤔 Thinking..."):
            try:
                show_cols = [
                    c for c in [
                        "projectname", "projecttype", "projectcategory",
                        "price", "cityid", "fulladdress", "aboutproperty"
                    ] if c in data.columns
                ]

                # Add a bit of property data for context
                context = data[show_cols].fillna("").head(10).to_string(index=False)

                prompt = (
                    f"User wants: {user_query}\n"
                    f"Here are some property listings:\n{context}\n\n"
                    f"Give a short, clear, helpful response based on the listings."
                )

                response = llm(prompt, max_new_tokens=150, temperature=0.7)
                reply = response[0]["generated_text"].strip()

                st.markdown("### 🤖 AI Response:")
                st.write(reply if reply else "Sorry, I couldn’t find anything relevant right now.")
            except Exception as e:
                st.error(f"Error generating response: {e}")

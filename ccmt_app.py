# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# st.set_page_config(layout="wide", page_title="CCMT Cutoff Explorer")
# st.title("🎓 CCMT M.Tech Cutoff Analysis & Prediction")

# # ----------------------------------
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#     "📁 Upload Data",
#     "🔍 Explore Data",
#     "📈 Trends",
#     "📉 Round Comparison",
#     "🏛️ 5-Year College View",
#     "🤖 Predict 2025 SR1–NSR"
# ])

# all_data = pd.DataFrame()

# # ----------------------------------
# with tab1:
#     st.header("Upload CCMT CSVs (2021–2025)")
#     uploaded_files = st.file_uploader("Upload all 5 files", type="csv", accept_multiple_files=True)

#     if uploaded_files:
#         for file in uploaded_files:
#             year = int(file.name.split('.')[0])
#             df = pd.read_csv(file, encoding='ISO-8859-1')
#             df.columns = [col.strip().replace("﻿", "") for col in df.columns]
#             df.rename(columns={
#                 'Rounds': 'Round',
#                 'College Name': 'Institute',
#                 'MTECH Course Name': 'PG Program',
#                 'S.No': 'Sr.No'
#             }, inplace=True)
#             df["Year"] = year
#             all_data = pd.concat([all_data, df], ignore_index=True)

#         st.success("✅ All files uploaded and processed successfully.")
#         st.dataframe(all_data.head())

# # ----------------------------------
# with tab2:
#     st.header("Explore Combined Dataset")
#     if not all_data.empty:
#         year = st.selectbox("Select Year", sorted(all_data["Year"].unique()))
#         df = all_data[all_data["Year"] == year]

#         inst = st.selectbox("Institute", sorted(df["Institute"].unique()))
#         valid_programs = sorted(df[df["Institute"] == inst]["PG Program"].unique())
#         prog = st.selectbox("Program", valid_programs)

#         rounds = sorted(df["Round"].unique())
#         cat = st.selectbox("Category", sorted(df["Category"].unique()))

#         filtered = df[
#             (df["Institute"] == inst) &
#             (df["PG Program"] == prog) &
#             (df["Category"] == cat)
#         ]

#         st.dataframe(filtered)

# # ----------------------------------
# with tab3:
#     st.header("Cutoff Trends Over Years")
#     if not all_data.empty:
#         inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="tr1")
#         valid_programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
#         prog = st.selectbox("Program", valid_programs, key="tr2")
#         cat = st.selectbox("Category", sorted(all_data["Category"].unique()), key="tr3")

#         df = all_data[
#             (all_data["Institute"] == inst) &
#             (all_data["PG Program"] == prog) &
#             (all_data["Category"] == cat)
#         ]

#         if not df.empty:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             sns.lineplot(data=df, x="Year", y="Min GATE Score", hue="Round", marker="o", ax=ax)
#             ax.set_title(f"Min GATE Score: {prog} at {inst} ({cat})")
#             st.pyplot(fig)

# # ----------------------------------
# with tab4:
#     st.header("Compare Rounds for Institute + Program")
#     if not all_data.empty:
#         inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="rc1")
#         valid_programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
#         prog = st.selectbox("Program", valid_programs, key="rc2")

#         df = all_data[(all_data["Institute"] == inst) & (all_data["PG Program"] == prog)]
#         if not df.empty:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             sns.boxplot(data=df, x="Round", y="Min GATE Score", hue="Category", ax=ax)
#             ax.set_title(f"Round Comparison for {prog} at {inst}")
#             st.pyplot(fig)

# # ----------------------------------
# with tab5:
#     st.header("5-Year History by Category for a Round")
#     if not all_data.empty:
#         inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="h1")
#         valid_programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
#         prog = st.selectbox("Program", valid_programs, key="h2")
#         rnd = st.selectbox("Round", sorted(all_data["Round"].unique()), key="h3")

#         df = all_data[
#             (all_data["Institute"] == inst) &
#             (all_data["PG Program"] == prog) &
#             (all_data["Round"] == rnd)
#         ]

#         pivot = df.pivot_table(index="Year", columns="Category", values="Min GATE Score")
#         st.dataframe(pivot)

#         if not pivot.empty:
#             fig, ax = plt.subplots(figsize=(10, 5))
#             pivot.plot(kind="bar", ax=ax)
#             ax.set_ylabel("Min GATE Score")
#             ax.set_title(f"5-Year Min GATE Score for {prog} ({rnd}) at {inst}")
#             st.pyplot(fig)

# # ----------------------------------
# with tab6:
#     st.header("Predict SR1, SR2, NSR for 2025")
#     if not all_data.empty:
#         train = all_data[all_data["Year"] < 2025]
#         base_2025 = all_data[
#             (all_data["Year"] == 2025) & 
#             (all_data["Round"].str.contains("R[123]", case=False))
#         ]

#         if not base_2025.empty:
#             df = pd.concat([train, base_2025], ignore_index=True)
#             label_cols = ["Institute", "PG Program", "Group", "Category", "Round"]
#             df[label_cols] = df[label_cols].astype(str)

#             dummy_rows = []
#             for r in ["SR1", "SR2", "NSR"]:
#                 row = df.iloc[0].copy()
#                 row["Round"] = r
#                 row["Year"] = 9999
#                 dummy_rows.append(row)
#             df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

#             encoders = {col: LabelEncoder().fit(df[col]) for col in label_cols}
#             for col in label_cols:
#                 df[col] = encoders[col].transform(df[col])

#             X_train = df[df["Year"] < 2025][["Year"] + label_cols]
#             y_train = df[df["Year"] < 2025]["Min GATE Score"]

#             model = RandomForestRegressor(n_estimators=200, random_state=42)
#             model.fit(X_train, y_train)

#             predictions = []
#             for _, row in base_2025.iterrows():
#                 for r in ["SR1", "SR2", "NSR"]:
#                     entry = row.copy()
#                     entry["Round"] = r
#                     entry["Year"] = 2025
#                     for col in label_cols:
#                         entry[col] = encoders[col].transform([str(entry[col])])[0]
#                     predictions.append(entry)

#             pred_df = pd.DataFrame(predictions)
#             pred_df["Min GATE Score"] = model.predict(pred_df[["Year"] + label_cols])

#             for col in label_cols:
#                 pred_df[col] = encoders[col].inverse_transform(pred_df[col])

#             result = pred_df[["Year", "Institute", "PG Program", "Group", "Category", "Round", "Min GATE Score"]]
#             st.dataframe(result)
#             st.download_button("📥 Download Predictions", result.to_csv(index=False), "predicted_2025.csv")
#         else:
#             st.warning("2025 R1–R3 data missing.")


# ccmt_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="CCMT M.Tech Cutoff Predictor")

st.title("🎓 CCMT M.Tech Cutoff Analysis & Prediction (2021–2025)")

tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs([
    "📁 Upload Data",
    "🔍 Explore Data",
    "📈 Trends",
    "📉 Round Comparison",
    "🏛️ 5-Year College View",
    "🤖 Predict 2025 SR1–NSR",
    "🤖 Predict 2025 SR1–NSR (college wise)"

])

# Global holder for all data
all_data = pd.DataFrame()

# 🟡 Round normalization function
def normalize_round(r):
    r = str(r).strip().upper()
    r = r.replace("ROUND", "R").replace(" ", "")
    return r

# ----------------------
# Tab 1: Upload
# ----------------------
# ----------------------
# Tab 1: Upload or Auto-Load Data
# ----------------------
with tab1:
    st.header("📁 Load Data (Auto or Manual)")

    @st.cache_data
    def load_csvs_from_root():
        expected_years = [2021, 2022, 2023, 2024, 2025]
        loaded_data = pd.DataFrame()
        loaded_years = []

        for year in expected_years:
            filename = f"{year}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename, encoding='ISO-8859-1')
                df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]

                # 🛠 Fix for varying column names
                col_map = {
                    "Rounds": "Round",
                    "S.No": "Sr.No",
                    "College Name": "Institute",
                    "MTECH Course Name": "PG Program"
                }
                df.rename(columns=col_map, inplace=True)

                df["Year"] = year
                df["Round"] = df["Round"].apply(normalize_round)
                loaded_data = pd.concat([loaded_data, df], ignore_index=True)
                loaded_years.append(year)

        return loaded_data, loaded_years

    all_data, loaded_years = load_csvs_from_root()

    if loaded_years:
        st.success(f"✅ Auto-loaded data from: {', '.join(map(str, loaded_years))}")
        st.dataframe(all_data.head())
    else:
        st.warning("⚠️ No files (2021.csv–2025.csv) found in project folder. Upload them manually:")

        uploaded_files = st.file_uploader("📤 Upload CSVs (2021–2025)", type="csv", accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                year = int(file.name.split('.')[0])
                df = pd.read_csv(file, encoding='ISO-8859-1')
                df.columns = [c.strip().replace("ï»¿", "") for c in df.columns]

                col_map = {
                    "Rounds": "Round",
                    "S.No": "Sr.No",
                    "College Name": "Institute",
                    "MTECH Course Name": "PG Program"
                }
                df.rename(columns=col_map, inplace=True)

                df["Year"] = year
                df["Round"] = df["Round"].apply(normalize_round)
                all_data = pd.concat([all_data, df], ignore_index=True)

            st.success("✅ Uploaded files loaded successfully.")
            st.dataframe(all_data.head())

# ----------------------
# Tab 2: Explore
# ----------------------
with tab2:
    st.header("Explore Combined Dataset")

    if not all_data.empty:
        year = st.selectbox("Year", sorted(all_data["Year"].unique()))
        df = all_data[all_data["Year"] == year]

        inst = st.selectbox("Institute", sorted(df["Institute"].unique()))
        valid_programs = sorted(df[df["Institute"] == inst]["PG Program"].unique())
        prog = st.selectbox("PG Program", valid_programs)
        rnd = st.selectbox("Round", ["All"] + sorted(df["Round"].unique()))
        cat = st.selectbox("Category", ["All"] + sorted(df["Category"].unique()))

        df = df[(df["Institute"] == inst) & (df["PG Program"] == prog)]
        if rnd != "All":
            df = df[df["Round"] == rnd]
        if cat != "All":
            df = df[df["Category"] == cat]

        st.dataframe(df)

# ----------------------
# Tab 3: Trends
# ----------------------
with tab3:
    st.header("Min GATE Score Trend Over Years")

    if not all_data.empty:
        inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="tr1")
        programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
        prog = st.selectbox("PG Program", programs, key="tr2")
        cat = st.selectbox("Category", sorted(all_data["Category"].unique()), key="tr3")

        df = all_data[
            (all_data["Institute"] == inst) &
            (all_data["PG Program"] == prog) &
            (all_data["Category"] == cat)
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x="Year", y="Min GATE Score", hue="Round", marker="o", ax=ax)
        ax.set_title(f"Min GATE Score Trend ({prog}, {inst}, {cat})")
        st.pyplot(fig)
        st.dataframe(df)
# ----------------------
# Tab 4: Round Comparison
# ----------------------
with tab4:
    st.header("Compare Rounds for One Institute + Program")

    if not all_data.empty:
        inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="rc1")
        programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
        prog = st.selectbox("PG Program", programs, key="rc2")

        df = all_data[(all_data["Institute"] == inst) & (all_data["PG Program"] == prog)]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="Round", y="Min GATE Score", hue="Category", ax=ax)
        ax.set_title(f"Round-wise Distribution ({prog}, {inst})")
        st.pyplot(fig)

# ----------------------
# Tab 5: College 5-Year View
# ----------------------
with tab5:
    st.header("5-Year Cutoff Table for One Round")

    if not all_data.empty:
        inst = st.selectbox("Institute", sorted(all_data["Institute"].unique()), key="v1")
        programs = sorted(all_data[all_data["Institute"] == inst]["PG Program"].unique())
        prog = st.selectbox("PG Program", programs, key="v2")
        rnd = st.selectbox("Round", sorted(all_data["Round"].unique()), key="v3")

        df = all_data[
            (all_data["Institute"] == inst) &
            (all_data["PG Program"] == prog) &
            (all_data["Round"] == rnd)
        ]

        pivot = df.pivot_table(index="Year", columns="Category", values="Min GATE Score")
        st.dataframe(pivot)

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Min GATE Score")
            ax.set_title(f"{prog} ({rnd}) - Category-wise 5-Year History ({inst})")
            st.pyplot(fig)

# ----------------------
# Tab 6: Prediction
# ----------------------
with tab6:
    st.header("Predict 2025 SR1 / SR2 / NSR")

    if not all_data.empty:
        train = all_data[all_data["Year"] < 2025]
        test_base = all_data[
            (all_data["Year"] == 2025) &
            (all_data["Round"].isin(["R1", "R2", "R3"]))
        ]

        if not test_base.empty:
            df = pd.concat([train, test_base], ignore_index=True)

            label_cols = ["Institute", "PG Program", "Group", "Category", "Round"]
            df[label_cols] = df[label_cols].astype(str)

            # Add dummy rows for unseen rounds
            dummy_rows = []
            for r in ["SR1", "SR2", "NSR"]:
                row = df.iloc[0].copy()
                row["Round"] = r
                row["Year"] = 9999
                dummy_rows.append(row)
            df_ext = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

            encoders = {col: LabelEncoder().fit(df_ext[col]) for col in label_cols}
            for col in label_cols:
                df[col] = encoders[col].transform(df[col])

            X = df[["Year"] + label_cols]
            y = df["Min GATE Score"]

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X[df["Year"] < 2025], y[df["Year"] < 2025])

            pred_rows = []
            for r in ["SR1", "SR2", "NSR"]:
                for _, base_row in test_base.iterrows():
                    row = base_row.copy()
                    row["Round"] = r
                    row["Year"] = 2025
                    pred_rows.append(row)

            pred_df = pd.DataFrame(pred_rows)
            for col in label_cols:
                pred_df[col] = encoders[col].transform(pred_df[col].astype(str))

            X_pred = pred_df[["Year"] + label_cols]
            pred_df["Predicted Min GATE Score"] = model.predict(X_pred)

            # Decode back for display
            for col in label_cols:
                pred_df[col] = encoders[col].inverse_transform(pred_df[col])

            final = pred_df[["Year", "Institute", "PG Program", "Group", "Category", "Round", "Predicted Min GATE Score"]]
            st.dataframe(final)

            st.download_button("📥 Download Prediction CSV", final.to_csv(index=False), file_name="2025_predictions.csv")
        else:
            st.warning("Upload 2025 data with rounds R1, R2, R3 first.")

# ----------------------
# Tab 6: Prediction (Updated)
# ----------------------
with tab7:
    st.header("🎯 Predict 2025 SR1 / SR2 / NSR (College-wise)")

    if not all_data.empty:
        available_colleges = sorted(all_data[all_data["Year"] == 2025]["Institute"].unique())
        selected_college = st.selectbox("Select Institute for Prediction", available_colleges)

        train = all_data[all_data["Year"] < 2025]
        test_base = all_data[
            (all_data["Year"] == 2025) &
            (all_data["Round"].isin(["R1", "R2", "R3"])) &
            (all_data["Institute"] == selected_college)
        ]

        if not test_base.empty:
            with st.spinner("Training model and generating predictions..."):
                df = pd.concat([train, test_base], ignore_index=True)

                label_cols = ["Institute", "PG Program", "Group", "Category", "Round"]
                df[label_cols] = df[label_cols].astype(str)

                # Add dummy rows to ensure label encoder handles SR1, SR2, NSR
                dummy_rows = []
                for r in ["SR1", "SR2", "NSR"]:
                    row = df.iloc[0].copy()
                    row["Round"] = r
                    row["Year"] = 9999
                    dummy_rows.append(row)
                df_ext = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

                encoders = {col: LabelEncoder().fit(df_ext[col]) for col in label_cols}
                for col in label_cols:
                    df[col] = encoders[col].transform(df[col])

                X = df[["Year"] + label_cols]
                y = df["Min GATE Score"]

                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X[df["Year"] < 2025], y[df["Year"] < 2025])

                pred_rows = []
                total = len(test_base) * 3  # 3 rounds for each row

                progress = st.progress(0)
                counter = 0

                for r in ["SR1", "SR2", "NSR"]:
                    for _, base_row in test_base.iterrows():
                        row = base_row.copy()
                        row["Round"] = r
                        row["Year"] = 2025
                        pred_rows.append(row)
                        counter += 1
                        progress.progress(counter / total)

                pred_df = pd.DataFrame(pred_rows)
                for col in label_cols:
                    pred_df[col] = encoders[col].transform(pred_df[col].astype(str))

                X_pred = pred_df[["Year"] + label_cols]
                pred_df["Predicted Min GATE Score"] = model.predict(X_pred)

                # Decode back for display
                for col in label_cols:
                    pred_df[col] = encoders[col].inverse_transform(pred_df[col])

                final = pred_df[["Year", "Institute", "PG Program", "Group", "Category", "Round", "Predicted Min GATE Score"]]
                st.success("✅ Prediction complete.")
                st.dataframe(final)

                st.download_button("📥 Download CSV", final.to_csv(index=False), file_name=f"{selected_college}_2025_predictions.csv")
        else:
            st.warning("⚠️ No R1–R3 data found for this college in 2025. Please upload that first.")

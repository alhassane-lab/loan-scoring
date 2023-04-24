import streamlit as st
import joblib
import pandas as pd
import os
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
import altair as alt
from PIL import Image
import numpy as np
import streamlit.components.v1 as components
import lime
from lime import lime_tabular
import streamlit.components.v1 as components


st.set_page_config(
    page_title="CREDIT SCORING - DACHBOARD CLIENT SCORING", page_icon="", layout="wide"
)


def load_file(path):
    model = joblib.load(open(os.path.join(path), "rb"))
    return model


@st.cache_data
@st.cache_resource
def load():
    loan_scoring_classifier = joblib.load(
        open(os.path.join("./models/best_model.pkl"), "rb")
    )
    training_features = joblib.load(
        open(os.path.join("./data/training_features.pkl"), "rb")
    )
    raw_data = joblib.load(open(os.path.join("./data/full_data.pkl"), "rb"))

    return loan_scoring_classifier, training_features, raw_data


loan_scoring_classifier, training_features, raw_data = load()

scaler = MinMaxScaler()
data = scaler.fit_transform(raw_data[training_features])
data = pd.DataFrame(data, index=raw_data.index, columns=training_features)
raw_data = raw_data.reset_index()
probas = loan_scoring_classifier.predict_proba(data)
raw_data["proba_true"] = probas[:, 0]
mean_score = raw_data["proba_true"].mean()

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(data),
    feature_names=data.columns,
    # class_names=['bad', 'good'],
    mode="classification",
)


def plot_preds_proba(customer_id):
    user_infos = {
        "Income": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_INCOME_TOTAL"
        ].values[0],
        "Credit": raw_data[raw_data["SK_ID_CURR"] == customer_id]["AMT_CREDIT"].values[
            0
        ],
        "Annuity": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_ANNUITY"
        ].values[0],
    }
    pred_proba_df = pd.DataFrame(
        {"Amount": user_infos.values(), "Operation": user_infos.keys()}
    )

    c = (
        alt.Chart(pred_proba_df)
        .mark_bar()
        .encode(x="Operation", y="Amount", color="Operation")
        .properties(width=330, height=310)
    )
    st.altair_chart(c)


def main():
    with st.sidebar:
        logo_image = Image.open("./data/logo.png")
        cover_image = Image.open("./data/cover.jpg")
        st.sidebar.image(cover_image, use_column_width=True)

        st.markdown(
            """
             <h4 style='text-align: center; color: black;'> Quick Tutorial </h4>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            """
            1. Select or edit **Customer ID**.
            2. Click **Show Infos** button.
            4. Analyze user informations and prediction results
            """,
        )
        st.sidebar.image(logo_image, use_column_width=True)

    st.title("Loan Scoring Model")

    with st.form(key="myform"):
        # user_liste = data.index
        # user_id_value = st.selectbox('Select customer id', user_liste)
        user_id_value = st.number_input("Select customer id", min_value=100001)
        submit_button = st.form_submit_button(label="Show")

        if submit_button:
            if isinstance(user_id_value, int) and user_id_value in data.index:
                st.write("Customer selected : ", user_id_value)
                col1, col2 = st.columns(2)
                # data = data.reset_index()
                user = data[data.index == int(user_id_value)]
                # prediction = loan_scoring_classifier.predict(user)[0]
                probas = loan_scoring_classifier.predict_proba(user)
                probabilities = dict(
                    zip(loan_scoring_classifier.classes_, np.round(probas[0], 3))
                )

                # display results
                with col1:
                    st.info("Customer Infos")
                    # st.text(f'User Id : {user_id_value}')
                    user_infos = raw_data[raw_data["SK_ID_CURR"] == user_id_value]

                    # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                    # st.write('Sex', user_infos["CODE_GENDER"].item())
                    # st.write('Status', user_infos["NAME_FAMILY_STATUS"].item())
                    # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                    # user_infos = user_infos[[]]
                    dict_infos = {
                        "Age": int(user_infos["DAYS_BIRTH"] / -365),
                        "Gender": user_infos["CODE_GENDER"]
                        .replace(["F", "M"], ["Female", "Male"])
                        .item(),
                        "Status": user_infos["NAME_FAMILY_STATUS"].item(),
                        "Education": user_infos["NAME_EDUCATION_TYPE"].item(),
                        "Employment_Seniority": int(
                            user_infos["DAYS_EMPLOYED"].values / -365
                        ),
                        "Income_Type": user_infos["NAME_INCOME_TYPE"].item(),
                        "Income_Ammount": user_infos["AMT_INCOME_TOTAL"].item(),
                        # "Type contrat" : user_infos["NAME_CONTRACT_TYPE"].item(),
                        # "Montant_credit" : user_infos["AMT_CREDIT"].item(),
                        # "Annuites" : user_infos["AMT_ANNUITY"].item(),
                    }
                    st.write(dict_infos)

                    # st.info('Results')
                    st.markdown("""---""")
                    st.info("Loan History")
                    dict_infos = {
                        "Type contrat": user_infos["NAME_CONTRACT_TYPE"].item(),
                        "Montant_credit": user_infos["AMT_CREDIT"].item(),
                        "Annuites": user_infos["AMT_ANNUITY"].item(),
                    }
                    user = data[data.index == int(user_id_value)]
                    st.write(dict_infos)
                    # st.metric(label='Accuracy', value='', delta='1.6')
                    st.markdown("""---""")
                    st.info("Credit Score")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    if round(probabilities[0] * 100, 2) > 60:
                        c3.metric(
                            "High Score",
                            value=round(probabilities[0] * 100, 2),
                            delta=f"{round((probabilities[0]-0.6)*100,2)}",
                        )

                        st.success("This customer is a potential refunder", icon="✅")
                    elif 50 < round(probabilities[0] * 100, 2) < 60:
                        c3.metric(
                            "Medium Score",
                            value=round(probabilities[0] * 100, 2),
                            delta=f"{round((probabilities[0]-0.6)*100,2)}",
                        )

                        st.warning(
                            "This customer may have difficulties in refunding",
                            icon="⚠️",
                        )
                    else:
                        c3.metric(
                            "Low Score",
                            value=round(probabilities[0] * 100, 2),
                            delta=f"{round((probabilities[0]-0.6)*100,2)}",
                        )

                        st.error("This customer can not refund", icon="🚨")
                    # c2.metric('Non refunder', value = probabilities[1], delta=f"{round(probabilities[1]*100,2)}%",delta_color="inverse")
                    # st.metric('Refunder', value = probabilities[0], delta=f"{round(probabilities[0]*100,2)}%")

                # dis
                with col2:
                    st.info("Features contribution")
                    exp = explainer.explain_instance(
                        data_row=data.loc[user_id_value],
                        predict_fn=loan_scoring_classifier.predict_proba,
                    )

                    # exp.show_in_notebook(show_table=True)

                    components.html(exp.as_html(), height=550)
                    st.markdown("""---""")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col2:
                        plot_preds_proba(user_id_value)
            else:
                st.error("Please, enter a valid customer id.", icon="🚨")

        else:
            st.markdown("""---""")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Project Lifecycle**")
                lifecycle_schema = Image.open("./data/lifecycle.png")
                st.image(lifecycle_schema, use_column_width=True)
            with col2:
                st.markdown("**Global Explainability**")
                explain_image = Image.open("./data/explainability.png")
                st.image(explain_image, use_column_width=True)

            HtmlFile = open("./data/data_drift_report.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()
            components.html(source_code, height=1800)


if __name__ == "__main__":
    main()

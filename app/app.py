import streamlit as st
import joblib
import pandas as pd
import os
from lightgbm import LGBMClassifier
import altair as alt
from PIL import Image
import numpy as np

st.set_page_config(page_title="CREDIT SCORING - DACHBOARD CLIENT SCORING", page_icon="", layout="wide")

def load_file(model_file):
    model = joblib.load(open(os.path.join(model_file), 'rb'))
    return model

@st.cache_data
@st.cache_resource
def load():
    loan_scoring_classifier = joblib.load(open(os.path.join('./models/best_lightGBM_model.pkl'), 'rb'))
    data = joblib.load(open(os.path.join('./data/raw_data_scaled_df.pkl'), 'rb'))  
    raw_data = joblib.load(open(os.path.join('./data/raw_data.pkl'), 'rb'))
    
    return loan_scoring_classifier, data, raw_data


loan_scoring_classifier, data, raw_data = load()
probas = loan_scoring_classifier.predict_proba(data)
raw_data['proba_true'] = probas[:,0]
mean_score = raw_data['proba_true'].mean()

def plot_preds_proba(customer_id):
    user = raw_data[raw_data.index==int(customer_id)]
    user_infos = {
        # "Age" : int(user["DAYS_BIRTH"]/ -365),
        # "Seniority" : int(user["DAYS_EMPLOYED"].values / -365),
        "Income" : user["AMT_INCOME_TOTAL"].item(),
        "Credit" : user["AMT_CREDIT"].item(),
        "Annuity" : user["AMT_ANNUITY"].item(),
    }
    pred_proba_df = pd.DataFrame({'Amount':user_infos.values(), 'Operation': user_infos.keys()})

    c =alt.Chart(pred_proba_df).mark_bar().encode(
        x='Operation',
        y='Amount',
        color="Operation"
    ).properties(
    width=330,
    height=330
)
    st.altair_chart(c)


    
def main():

    with st.sidebar:
        logo_image = Image.open("./data/logo.png") 
        cover_image = Image.open("./data/cover.jpg")        
        st.sidebar.image(cover_image,use_column_width=True)

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
        st.sidebar.image(logo_image,use_column_width=True)

    

    st.title('Loan Scoring Model')
    
    with st.form(key='myform'):
        #user_liste = data.index.astype(np.int32)
        #user_id_value = st.selectbox('Select customer id', user_liste)
        user_id_value = st.number_input('Select customer id', min_value=100001)
        submit_button = st.form_submit_button(label='Show')
        
        if submit_button:
            if isinstance(user_id_value, int) and user_id_value in data.index:
                st.write('Customer selected : ', user_id_value)
                col1, col2 = st.columns(2)
                #data = data.reset_index()
                user = data[data.index==int(user_id_value)]
                prediction = loan_scoring_classifier.predict(user)[0]
                probas = loan_scoring_classifier.predict_proba(user)
                probabilities = dict(zip(loan_scoring_classifier.classes_, np.round(probas[0], 3)))

                
                # display results
                with col1:
                    st.info('Customer Infos')
                    #st.text(f'User Id : {user_id_value}')
                    user_infos = raw_data[raw_data['SK_ID_CURR']==user_id_value]
                    # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                    # st.write('Sex', user_infos["CODE_GENDER"].item())
                    # st.write('Status', user_infos["NAME_FAMILY_STATUS"].item())
                    # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                    #user_infos = user_infos[[]]
                    dict_infos = {
                            "Age" : int(user_infos["DAYS_BIRTH"]/ -365),
                            "Gender" : user_infos["CODE_GENDER"].replace(['F', 'M'], ['Female','Male']).item(),
                            "Status" : user_infos["NAME_FAMILY_STATUS"].item(),
                            "Education" : user_infos["NAME_EDUCATION_TYPE"].item(),
                            "Employment_Seniority" : int(user_infos["DAYS_EMPLOYED"].values / -365),
                            "Income_Type":user_infos["NAME_INCOME_TYPE"].item(),
                            "Income_Ammount" : user_infos["AMT_INCOME_TOTAL"].item(),
                            # "Type contrat" : user_infos["NAME_CONTRACT_TYPE"].item(),
                            # "Montant_credit" : user_infos["AMT_CREDIT"].item(),
                            # "Annuites" : user_infos["AMT_ANNUITY"].item(),
                        }
                    st.write(dict_infos)
                    
                    #st.info('Results')
                    
                    #st.metric(label='Accuracy', value='', delta='1.6')
                    st.info('Credit Score')
                    c1, c2, c3 = st.columns(3)
                    if round(probabilities[0]*100,2) > 70:
                        c2.metric('High Score', value = round(probabilities[0]*100,2), delta=f"{round((probabilities[0]-0.7)*100,2)}")
                        st.success('This customer is a potential refunder',icon="✅")
                    elif 50 < round(probabilities[0]*100,2) < 70:
                        c2.metric('Medium Score', value = round(probabilities[0]*100,2), delta=f"{round((probabilities[0]-0.7)*100,2)}")
                        st.warning('This customer may have difficulties in refunding', icon="⚠️") 
                    else:
                        c2.metric('Low Score', value = round(probabilities[0]*100,2), delta=f"{round((probabilities[0]-0.7)*100,2)}")
                        st.error('This customer can not refund', icon="🚨")
                    #c2.metric('Non refunder', value = probabilities[1], delta=f"{round(probabilities[1]*100,2)}%",delta_color="inverse")
                    #st.metric('Refunder', value = probabilities[0], delta=f"{round(probabilities[0]*100,2)}%")
                    
                    
                    
                # dis
                with col2:
                    st.info('Loan History')
                    dict_infos = {
                            "Type contrat" : user_infos["NAME_CONTRACT_TYPE"].item(),
                            "Montant_credit" : user_infos["AMT_CREDIT"].item(),
                            "Annuites" : user_infos["AMT_ANNUITY"].item(),
                        }
                    user = data[data.index==int(user_id_value)]
                    st.write(dict_infos)
                    st.markdown("""---""")
                    plot_preds_proba(user_id_value)


                    # #st.metric(label='Accuracy', value='', delta='1.6')
                    # c1, c2 = st.columns(2)
                    # c1.metric('Refunder', value = probabilities[0], delta=f"{round(probabilities[0]*100,2)}%")
                    # c2.metric('Non refunder', value = probabilities[1], delta=f"{round(probabilities[1]*100,2)}%",delta_color="inverse")
            else:
                st.error('Please, enter a valid customer id.', icon="🚨")
                    
  
    

if __name__ == '__main__':
	main()
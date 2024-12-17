
import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('final_model_xgb.pkl','rb') as file:
    model = pickle.load(file)

with open('transformer.pkl','rb') as file:
        pt = pickle.load(file)

    
def prediction(input_list):

    tran_data = pt.transform([[input_list[0],input_list[3]]])
    input_list[0] = tran_data[0][0]
    input_list[3] = tran_data[0][1]

    input_list = np.array(input_list,dtype=object)
      
    pred = model.predict_proba([input_list])[:,1][0]
    
    if pred>0.5:
        return f'This booking is more likely to get canceled: Chances {round(pred,2)}'
    else:
        return f'This booking is less likely to get canceled: Chances {round(pred,2)}'
    
def main():
    st.title('INN HOTEL GROUP')
    lt = st.text_input('Enter the lead time.')
    mst = (lambda x:1 if x=='Online' else 0)(st.selectbox('Enter the type of booking',['Online','Offline']))
    spcl = st.selectbox('Select the no of special requests made',[0,1,2,3,4,5])
    price = st.text_input('Enter the price offered for the room')
    adult = st.selectbox('Select the no adults in booking',[0,1,2,3,4])
    wkd = st.text_input('Enter the weekend nights in the booking')
    wk = st.text_input('Enter the week nights in booking')
    park = (lambda x:1 if x=='Yes'else 0)(st.selectbox('Is parking included in the booking',['Yes','No']))
    month = st.slider('What will be month of arrival',min_value=1,max_value=12,step=1)
    day = st.slider('What will be day of arrival',min_value=1,max_value=31,step=1)
    wkday_lambda = (lambda x: 0 if x=='Mon' else 1 if x=='Tue' else 2 if x=='Wed' else 3 if x=='Thus' 
                    else 4 if x=='Fri' else 5 if x=='Sat' else 6)
    wkday = wkday_lambda(st.selectbox('What is the weekday of arrival',['Mon','Tue','Wed','Thus','Fri','Sat','Sun']))

    
    inp_list = [lt,mst,spcl,price,adult,wkd,park,wk,month,day,wkday]
    
    if st.button('Predict'):
        response = prediction(inp_list)
        st.success(response)
        
if __name__=='__main__':
    main()
    

import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('final_model_xgb.pkl','wb') as file:
    model=pickle.load(file)

with open('transformer.pkl','wb') as file:
    pt=pickle.load(file)

# [lt_t,mst,spcl,price_t,adults,wkd,park,wk,month,day,weekd]
def prediction(input_list):
    input_list=np.array(input_list,dtype=object)
    pred=model.predict(input_list)
    pred_prob=model.predict_proba([input_list])[:,1][0] # probability of 1
    if pred_prob>0.5:
        return f'This booking is more likely to get canceled : Chances {round(pred_prob,2)}'
    else :
        return f'This booking is less likely to get canceled : Chances {round(pred_prob,2)}'
    
def main():
    st.title('INN HOTEL GROUP')
    lt=st.text_input('Enter the Lead time.')
    mst=(lambda x:1 if x=='Online' else 0)(st.selectbox('Enter the type of Booking',['Online','Offline']))
    spcl=st.selectbox('Select the number of Special Request made',[0,1,2,3,4,5])
    price=st.text_input('Enter the Price offered for the room')
    adults=st.selectbox('Select the Total Number of Adults in Booking'[0,1,2,3,4])
    wkd=st.text_input('Enter the Weekend Nights in the Booking')
    wk=st.text_input('Enter the Week Nights in the booking')
    park=(lambda x:1 if x=='Yes' else 0)(st.selectbox('Is parking included in the booking',['Yes','No']))
    month=st.slider('What will be the Month of Arival',min_value=1,max_value=12,step=1)
    day=st.slider('What will be the Day of Arival?',min_value=1,max_value=31,step=1)
    wkday_lambda=(lambda x:0 if x=='Mon' else 1 if x=='Tue' else 2 if x=='Wed' else 3 if x=='Thrus' else 4 if x=='Fri' else 5 if x=='Sat' else 6)
    wkday=wkday_lambda(st.selectbox('What is the Weekday of arival?'['Mon','Tue','Wed','Thrus','Fri','Sat','Sun']))
    tran_data=pt.transform([[float(lt),float(price)]])[0]
    lt_t=tran_data[0]
    price_t=tran_data[1]

    inp_list=[lt_t,mst,spcl,price_t,adults,wkd,park,wk,month,day,wkday]

    if st.button('Predict'):
        response=prediction(inp_list)
        st.success(response)


if __name__=='__main__':
    main()

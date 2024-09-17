import numpy as np
import seaborn as sns
from PIL import Image
import time
import pickle

import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import json
import streamlit as st
import time
import matplotlib.pyplot as plt



###############################################--------#####################################################
files={"Random Forest":"rf_pkl","Decision Tree":"dt_pkl","Naive Bayes":"naive_pkl","Logistic Regression":"lr_pkl"}
scores={"Random Forest":0.85,"Decision Tree": 0.672,"Naive Bayes":0.816,"Logistic Regression":0.805}
count=0

#animation image
def load_lottiefile(img:str):
    with open(img,"r") as f:
        return json.load(f)
lot=load_lottiefile(r"json\tree.json") 
lot1=load_lottiefile(r"json\\emoji.json") 
lot2=load_lottiefile(r"json\\sad.json") 
#
model=load_model("deep_learning\PotatoDiseaseClassification.keras")
data_cat=['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']
st.header("Image Classification 🌿")

img=st.file_uploader("Upload an image ",type=['jpg','jpeg','png'])
col1,col2=st.columns(2)
with col1:
    if img:
        with st.sidebar:
            st_lottie(lot,height=70)
        

       
        with col2:
            
            alg=st.radio("Select",("None",'Deep learning Algorithms','Machine learning Algoritmns'))
            
            if alg is "Deep learning Algorithms":
                
                select=st.selectbox("select",["","CNN"])
                if select=="CNN":
                    img=tf.keras.utils.load_img(img,target_size=(256,256))
                    img_arr=tf.keras.utils.array_to_img(img)
                    img_bat=tf.expand_dims(img_arr,0)

                    predict=model.predict(img_bat)
                    score=tf.nn.softmax(predict)
                
              
            
            elif alg is 'Machine learning Algoritmns':
                select=st.selectbox("Select",files.keys())
                if(select =="Naive Bayes" or select=="Logistic Regression" or select=="Random Forest" or select=="Decision Tree"):
                    im=Image.open(img)
                    with open("machine_learning_models"+"\\"+str(files[select]), 'rb') as f:

                        mv= pickle.load(f)
                        prediction=mv.predict(np.asarray(im).reshape(1,-1))
                else:
                    pass
                        
            
               






                    
                    

        st.image(img)
        bt=st.button("Predict")        

        st.sidebar.title("Hello Welcome To Our Project")
        name=st.sidebar.text_input("Enter your name")
        
        num=st.sidebar.text_input("Enter your last 4 digit phone number")
        
        for n, i in enumerate(name):
            
            if((ord(i)==77 and  n==0)   or (ord(i)==109 and n==0)):
                count=count+1
            elif((ord(i)==65 and  n==1) or (ord(i)==97 and  n==1)):
                count=count+1
            elif((ord(i)==78 and  n==2) or (ord(i)==110 and  n==2)):
                count=count+1
            elif((ord(i)==73 and  n==3) or (ord(i)==105 and  n==3)):
                count=count+1
            

            #
            elif((ord(i)==83 and  n==0)   or (ord(i)==115 and n==0)):
                count=count+1
            elif((ord(i)==87 and  n==1) or (ord(i)==119 and  n==1)):
                count=count+1
            elif((ord(i)== 65 and  n==2) or (ord(i)==97 and  n==2)):
                count=count+1
            elif((ord(i)==84 and  n==3) or (ord(i)==116 and  n==3)):
                count=count+1

            elif((ord(i)== 72 and  n==4) or (ord(i)==104 and  n==4)):
                count=count+1
            elif((ord(i)==73 and  n==5) or (ord(i)==105 and  n==5)):
                count=count+1
            elif(ord(i)>0):
                    count=count+1
            
           
        
            
        
       
                
                
                        
        b=st.sidebar.button("Login")
        if((count==4 and num=="4726") or (count==6 and num=="5272")):
            if  b:
                st.sidebar.header(" 😊")
                st.sidebar.header("You are the right person to predict")
                

            elif(bt and alg=="Machine learning Algoritmns"):
                st.caption(select)
                st.success("Potato "+prediction[0])
                for i in files.keys():
                    if(select==i):
                        st.progress(int(scores[i]*100))
                        st.success("Accuracy of "+i+" "+str(scores[i]*100))


                
            elif bt :
                
                st.sidebar.header(" 😊")
                st.sidebar.header("You are the right person to predict")
                st.success("leaf in images shows "+data_cat[np.argmax(score)])
                acc=np.round(np.max(score)*100,2)
                st.progress(int(acc))
                st.success("Accuracy "+str(acc))

                
        elif b:
            st.sidebar.header("Sorry You don't have authority to predict👎")
            with st.sidebar:
                st_lottie(lot1,height=200)
            
            

        if((bt and count==6 and  num=="5272") or (bt and count == 4 and num=="4726")):
            pass
        elif bt :
            st.warning("Please Enter your correct Credentials and login then select Algorithm")
            with st.sidebar:
                st_lottie(lot1,height=200)
            st_lottie(lot2,height=200)
    
   


    
    
        
        

        


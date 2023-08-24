import streamlit as st
import pandas as pd
import numpy as np

import pickle

model=pickle.load(open('/Users/ambujasenapati/Song Predictor/trained_model.sav','rb'))


def main():
        st.title('Making popularity prediction')
        st.markdown('input the values:')

        st.header("Song features")

        col5,col9 = st.columns(2)
        col10,col7 =  st.columns(2)
        col13,col14,col15,col16 =  st.columns(4)
        col4,col18,col19,col20 =  st.columns(4)
        col17,col21,col6 = st.columns(3)
            

                

                                    
        with col4:

            duration_ms = st.number_input('duration_ms')
                    
        with col5:
            explicit = st.number_input('explicit')
                    
        with col6:
            time_signature = st.number_input('time_signature')


                    
        with col9:
            loudness = st.number_input('loudness')
                        

                    
        with col10:
            mode = st.number_input('mode')
                    

                    


                    
        with col7:
            release_month = st.selectbox('Month:',['January', 'February', 'March', 'April', "May",'June','July','August','September','October','November','December'])
            if release_month == 'January':
                release_month=1

            elif release_month == 'February':
                release_month=2

            elif release_month == 'March':
                release_month =3
            elif release_month == 'April':
                release_month = 4
            elif release_month == 'May':
                release_month=5
            elif release_month == 'June':
                release_month =6

            elif release_month == 'July':
                release_month = 7

            elif release_month == 'August':
                release_month=8
            elif release_month == 'September':
                release_month = 9
            elif release_month == 'October':
                release_month = 10
            elif release_month == 'November':
                release_month= 11
            elif release_month == 'December':
                release_month = 12 
                

                    
                    
        with col13:

            danceability  = st.number_input('danceability')
                        
        with col14:
            energy = st.number_input('energy')
                    
        with col15:
            key = st.number_input('key')
                    
        with col16:
            speechiness = st.number_input('speechiness')
                        
        with col17:
            acousticness = st.number_input('acousticness')
                    
        with col18:
            instrumentalness = st.number_input('instrumentalness')
                        
        with col19:
            liveness = st.number_input('liveness')
                    
        with col20:
            valence   = st.number_input('valence  ')
                                        

        with col21:
            tempo = st.number_input('tempo')
                


      
        input_data= [duration_ms,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,time_signature,release_month]
        input_arr=np.asarray(input_data).reshape(1,-1)


        if st.button("predict"):

            result = model.predict(input_arr)
            output = round(result[0],2) 
            
            st.success(output)     
        # result = predict(duration_ms,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,time_signature,release_month)
        # st.success(f'The predicted popularity of the song is {result[0]:.2f}')
if __name__=='__main__':
    main()




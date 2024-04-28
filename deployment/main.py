#
#
#
import pandas as pd
import pandas as pd
import streamlit as st
import base64
from MFCC_feature import getMFCC, getadiuo,getgreater
from keras.models import load_model
import numpy
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
# from bokeh.models.widgets import Button
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events
#navicon and header




def main():
    st.set_page_config(page_title="Deep Fake Detection", page_icon="🐸", layout="wide")  
        #  # load CSS Style
    with open('styles.css')as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://tailwindui.com/img/beams-home@95.jpg");
    background-size: 150%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    </style>
    """
#data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBw0KCAgICAgHBwgHBwoHBwcHBw8IDQcKFREWFhURExMYHSggGBoxGxUTITEhJSkrLi4uFx8zOD84NygtLisBCgoKDQ0NDg0NDjcZFRkrLSsrLSstLSstLTctKy0rLS0tKysrKystKzcrKysrKysrKysrKysrNysrKy0tKysrK//AABEIAHkBoQMBIgACEQEDEQH/xAAaAAEBAQEBAQEAAAAAAAAAAAAAAgEDBAUH/8QAGhABAQEBAQEBAAAAAAAAAAAAAAECEQMxIf/EABgBAQEBAQEAAAAAAAAAAAAAAAABAgME/8QAGhEBAQEBAQEBAAAAAAAAAAAAAAERAkEDIf/aAAwDAQACEQMRAD8A/cBoCWWKYoixNjpYyxdRw1lFy9FiLlqUebWUXL0XKNZblXXnsTY7XKLlWpXMVYng3KqVea5KlSxrXeVccc10lZSxXGWNjSVzsctZcd4eqxGsunPTl1zrw6zxXnvld94cdYdd15+vnZ+x7vD2+PVL18jz1yvd4erh3x7HT59+V6bE3Kpetsc3Zw1lz1l6LEay1KPLrKLl6dZc7l0nSPPcp473KLluVMc+Lhw4ai810zXKOkZqu2a6ZrhmusrnYrtKpyzVyudirbEtiKoBAAAAAAAAAAAY0BjLGiiLGWL4zijlcouXaxNiyo4XLncvTcudy3KPPcpsd7lFy1rWuPB0uU2DUrIvNQqJY1rrKuOOa6Ss4lWyxsaMVz1ly1h6bEXLU6ZsePWFeeuV21hy1l03XDrj2PX4+r0y9fNxePX5ejl3y1x15XdNipetc3ZxuUXLvYmxqVHmuUXL0XKLluVHC5Tx2uU2NaIVDjeKjY6ZrnFRmq7SrlcpVysVXWNRKqMUVKpEUitAQAAAAAAAAAAGNAYcaxRPGcWzgIsRY68ZYuo43KLl3sTctSjz3KLl6LlFy1KrhYzjtcouWl1MXKnjYlXXSVccouVmiywikZc7lGsu/E3LUqPNrJi8dtZc7lvdc7y9Hlt3leHN49Hntz65Xmu3GWNlLGHRFiLl1sZYsqOFym5drE2NSjjYzjrYmxrURxsbw4DcriIqJR0lXK5xUZqrjYmKZVTUxqDQEAAAAAAAAAAAAGDQGM40UTxNi+M4o52JuXXjLF1HC5Tcu9iblrRwsZx1uU2LqoVDjRVRURFRlFw4RqCbEXLrxli6jhcmbx1uUXLWs2OuNOsrzZ/HbGmOosrozjZRlpNibF8ZxdHO5TY62Jsa1HLhx0sTxdE8G8aBFRMVEFRURFRBUbEtjKqaxqAAAAAAAAAAAAAAAxoDBrAZxnFCiOMsXxnF0c7E3Lrxli6jjcs462JsXRHGxvDgrYqJbEFHBoieMsWcNHK5Iuxli6LzpcrjHTNZsFs42CKnjOK4zgJ4mx0ZxrRz4cXxnDUSKFAg1BsbGQRVNS2IKGNQAAAAAAAAAAAAAAAAGNAYNYozjOKARxnF8ZxdEWM46cZw1HPjVcOKMjRsQAbwGcZYrhwEcIpnAVKqIVKlVTCNQTxnFCiWK4cBPGcVwUS0ABoA0EGtY1AAAAAAAAAAAAAAAAAAAAAY0Bg1gM4NFE8OKATwacBjRoMGgMZxQCeEUwGtS0GgIMGgM4zimAzg0UYNAY0agxoAAAAAAAAAAAAAAAAAAAAAAAAAAAMaAwaAwBQBqAAAxoDBoAAAAAAAAAAAAAAAAD//2Q==    
#https://tailwindui.com/img/beams-home@95.jpg
#https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfPB9NZIWqsDh02tYlBODcgPCmUtdAQeU1WyK6r805MJNmijZj7PMMH-XPLHRN7VQj0p0&usqp=CAU

    model = load_model('models//old but gold last.h5')
    model_pro1= load_model('models//nn.h5')
    model_pro2= load_model('models//modelcnn.h5')
    model_pro3= load_model('models//last_pro5.h5')
    pro=[]

    st.markdown(page_bg_img, unsafe_allow_html=True)



    with st.container():
        st.image("compants\\Frame 10.png",caption="")



    


    st.markdown('<img class="line" src="https://firebasestorage.googleapis.com/v0/b/yalla-7324f.appspot.com/o/Line%202.png?alt=media&token=0c85831c-7741-4843-8fde-bbc205c5ec99" />', unsafe_allow_html=True)

    col1, col2 = st.columns([2.5,1.5])
    with col1:
        st.markdown( '<div class="audio-guardian-unmask-the-truth"> <span> <span class="audio-guardian-unmask-the-truth-span">Audio Guardian</span> <span class="audio-guardian-unmask-the-truth-span2">, Unmask The Truth</span></span></div>', unsafe_allow_html=True)
        st.markdown(' <div class="don-t-fall-victim-to-deepfake-scams"> Don\'t Fall Victim to Deepfake Scams! </div> <img class="aac-2040926-1" src="https://firebasestorage.googleapis.com/v0/b/yalla-7324f.appspot.com/o/aac-2040926-1.png?alt=media&token=62819211-e1d4-4d9a-8122-25e85b5255b9" />', unsafe_allow_html=True)
        st.markdown('<div class="probabilities">Probabilities</div>',unsafe_allow_html=True)
        real = st.empty()
        deepfake = st.empty()
     



        ############################
        # col3, col4 = st.columns(2)
        # with col3:
        
        #  deepfake_prob = st.empty()
        # with col4:
        #  real_prob = st.empty()
        # #######################
  


    with col2:
        uploaded_file = st.file_uploader("", type=["wav"])
        col5, col6, col7 = st.columns([2,1.5,1.5])
        Deepfake_bt=col6.button("Detect Deepfake")
  





        waveform_container = st.empty()
        st.markdown( '<div class="Waveform">Waveform</div>',unsafe_allow_html=True)
        image_container = st.empty()
    if uploaded_file is not None :
        # Process audio file using appropriate libraries
        # ...
        # Perform deepfake detection (if applicable)
        # ...
        # Display results
        
        audio=getadiuo(uploaded_file)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(audio, color="#76b3e8")
        ax.axis("off")
        fig.patch.set_alpha(0)  # Set figure background to transparent
        ax.set_facecolor("none")  # Set axis background to transparent
        # Display the plot in Streamlit
    
        image_container.pyplot(fig)

        waveform_container.audio(uploaded_file)



















    if (uploaded_file is not None) and Deepfake_bt:


        
        mfcc=getMFCC(uploaded_file)
        results=model.predict(mfcc).argmax(axis=1)
        
        pro0=model_pro1.predict(mfcc)
        
        print("m1",pro0)
        pro_1=model_pro2.predict(mfcc)
        print("m2",pro_1)
        pro_2=model_pro3.predict(mfcc)
        print("m3",pro_2)
        voting=(pro0+pro_2+results)/3

        pro_real=voting[0][1]
        pro_fake=voting[0][0]
        if results == 1:

            # deepfake.text(f"this is Real : ")
            print(voting.argmax(axis=1))
               
            if(voting.argmax(axis=1)!=1):
                print("rrrrrrrrrrrrrrrrrrrrr")
                pro_real,pro_fake=getgreater(pro_real,pro_fake)

            real.markdown(f'<div class="real_continar"><span> <span class="real"> This is Real </span> <span class="fake-probability-continar"><span class="high-fake-probability-strong-likelihood-of-artificial-origin">\nFake with :</span> <span class="high-fake11-probability-strong-likelihood-of-artificial-origin">{round(pro_fake*100,2)}%</span></span>\n     <span class="real-probability-continar"> <span class="high-real-probability-strong-indication-of-authenticity">Real with :</span> <span class="high-real11-probability-strong-indication-of-authenticity">{round(pro_real*100,2)}%</span></span></span></div>', unsafe_allow_html=True)


            # st.markdown('<div class="real">this is Real :</div>', unsafe_allow_html=True)
            # # If you want to keep the deepfake.text functionality
            # real.text("")
            # deepfake_prob = st.empty()
            # #with col4:
    
        

            
            # real_prob = st.empty()

            # st.markdown('<div class="high-fake-probability-strong-likelihood-of-artificial-origin">Fake with :</div>',unsafe_allow_html=True)
            
            # # Update the text position using the custom class

            #  <span> <span class="audio-guardian-unmask-the-truth-span">Audio Guardian</span> <span class="audio-guardian-unmask-the-truth-span2">, Unmask The Truth</span></span>
            # deepfake_prob.markdown(f'<div class="high-fake11-probability-strong-likelihood-of-artificial-origin"><{round(v[0][0]*100,2)}%</div>', unsafe_allow_html=True)
            # st.markdown('<div class="high-real-probability-strong-indication-of-authenticity">Real with :</div>',unsafe_allow_html=True)  
            # # Custom CSS style to change the position
        
            # # Update the text position using the custom class
            # real_prob.markdown(f'<div class="high-real11-probability-strong-indication-of-authenticity">{(v[0][1]*100,2)}%</div>', unsafe_allow_html=True)





        else:
            # real.text(f"this is Fake :")
            if(voting.argmax(axis=1)!=0):
                print("FFFFFFFFFFFFFFFFFFFF")
                pro_fake,pro_real=getgreater(pro_real,pro_fake)



            print('F',voting.argmax(axis=1))
            # Display waveform
            deepfake.markdown(f'<div class="real_continar"><span> <span class="real"> This is Fake </span> <span class="fake-probability-continar"><span class="high-fake-probability-strong-likelihood-of-artificial-origin">\nFake with :</span> <span class="high-fake11-probability-strong-likelihood-of-artificial-origin">{round(pro_fake*100,2)}%</span></span>\n     <span class="real-probability-continar"> <span class="high-real-probability-strong-indication-of-authenticity">Real with :</span> <span class="high-real11-probability-strong-indication-of-authenticity">{round(pro_real*100,2)}%</span></span></span></div>', unsafe_allow_html=True)

            # #with col4:
    
            # real_prob = st.empty()
            # st.markdown('<div class="deepfake">this is Fake :</div>', unsafe_allow_html=True)
            # # If you want to keep the real.text functionality
            # deepfake.text("")


            # st.markdown('<div class="high-fake-probability-strong-likelihood-of-artificial-origin">Fake with :</div>',unsafe_allow_html=True)
            # # Custom CSS style to change the position

            # # Update the text position using the custom class
            # deepfake_prob.markdown(f'<div class="high-fake11-probability-strong-likelihood-of-artificial-origin">{round(v[0][0]*100,2)}%</div>', unsafe_allow_html=True)


            # st.markdown('<div class="high-real-probability-strong-indication-of-authenticity">Real with :</div>',unsafe_allow_html=True)  
            # # Custom CSS style to change the position

            # # Update the text position using the custom class
            # real_prob.markdown(f'<div class="high-real11-probability-strong-indication-of-authenticity">{(v[0][1]*100,2)}%</div>', unsafe_allow_html=True)




        # deepfake_prob.text(f"Rael with :{round(v[0][1]*100,2)} %")    
        # real_prob.text(f"Fake with:{round(v[0][0]*100,2)}%")   




    #st.write(mfcc)






   
    
   


 








    # stt_button = Button(label="Speak", width=100)

    # stt_button.js_on_event("button_click", CustomJS(code="""
    #     var recognition = new webkitSpeechRecognition();
    #     recognition.continuous = true;
    #     recognition.interimResults = true;
    
    #     recognition.onresult = function (e) {
    #         var value = "";
    #         for (var i = e.resultIndex; i < e.results.length; ++i) {
    #             if (e.results[i].isFinal) {
    #                 value += e.results[i][0].transcript;
    #             }
    #         }
    #         if ( value != "") {
    #             document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
    #         }
    #     }
    #     recognition.start();
    #     """))

    # result = streamlit_bokeh_events(
    #     stt_button,
    #     events="GET_TEXT",
    #     key="listen",
    #     refresh_on_update=False,
    #     override_height=75,
    #     debounce_time=0)

    # if result:
    #     if "GET_TEXT" in result:
    #         st.write(result.get("GET_TEXT"))



    # st.markdown("""
    # <style>
    # body {
    #     font-family: sans-serif;
    # }
    # </style>
    # """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
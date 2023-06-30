import streamlit as st
import util
from PIL import Image

footer = """
<style>
    footer{
        visiblity: visible;
        
        }
    footer:after{
        content: "@Priyanshu";
        display: block;
        color: white;
        font-size: 20px;
        font-family: cursive;
        position: relative;
        # text-align: right;
        padding-right: 7px;
        padding-bottom: 10px;
        }
"""

st.set_page_config(page_title="Sportsperson Image Classifier", layout="wide")
st.markdown(footer, unsafe_allow_html=True)
st.title("SportsPerson Image Classifier")
st.markdown("""---""")
cr7, lm, vk, sm, ney = st.columns(5)

with cr7:
    image = Image.open('./imgs/cr7.jpg')
    image = image.resize((700, 750))
    st.image(image, caption="Cristiano Ronaldo")
with lm:
    image = Image.open('./imgs/lm.jpg')
    image = image.resize((700, 750))
    st.image(image, caption="Lionel Messi")

with ney:
    image = Image.open('./imgs/neymar.jpg')
    image = image.resize((700, 750))
    st.image(image, caption="Neymar Jr.")

with sm:
    image = Image.open('./imgs/sm.jpg')
    image = image.resize((700, 750))
    st.image(image, caption="Smriti Mandhana")

with vk:
    image = Image.open('./imgs/vk.png')
    image = image.resize((700, 750))
    st.image(image, caption="Virat Kohli")
st.markdown("""---""")
upload, display_result = st.columns(spec=[0.45, 0.55], gap="large")

a = """<style>
            .css-q8sbsg p, .css-9ycgxx {
                    font-size: 18px;
                    font-weight: bold;
                }
    """

with upload:
    st.markdown(a, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    img = util.load_file(uploaded_file)
with display_result:
    if img is not None:
        util.classify_img(img)

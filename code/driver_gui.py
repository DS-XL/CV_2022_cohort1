from define import *
from preprocessing import *
from engine import get_recommend
from utility import retrieve_all_image

from PIL import Image
import streamlit as st


# define the image source dir
IMAGE_DIR = "./datasets/cv2"


# define the following constants for the streamlit app components

st.header("Meet Fresh Recommendation Engine")

# sidebar for the user's input
st.sidebar.header("User Input Parameters")

subTxt = "Please complete the survey in order to get a recommendation(s). \n\n Note, we do not share or store your personal information with any third parties! All of your input will be cleared once the broswer tab is closed!."
st.sidebar.write(subTxt)

# sidebar options
opt_gender = st.sidebar.selectbox("Gender", GENDER)
opt_age = st.sidebar.selectbox("Age", AGE)
opt_eth = st.sidebar.selectbox("Ethnicity", ETHNICITY)
opt_sensOnPrice = st.sidebar.select_slider(
    'Sensitivity on Price (in the scale of 1 - 5)', SENSONPRICE)
opt_sensOnK = st.sidebar.select_slider(
    'Sensitivity on Calorie (in the scale of 1 - 5)', SENSONPRICE)
opt_sensSeasonalFood = st.sidebar.select_slider(
    'Sensitivity on Seasonal Food (in the scale of 1 - 5)', SENSONPRICE)
opt_HorK = st.sidebar.selectbox("Hot or Cold", HORC)
opt_allergy = st.sidebar.multiselect("Allergy (Select Multiple if Apply)",
                                     ALLERGY,
                                     default=None)
btnSubmit = st.sidebar.button("Submit")
st.sidebar.write("\n")

# output user's feature matrix
# st.write("Your feature matrix is:")
# st.write("Your gender: ", enc_gender(opt_gender))
# st.write("You age: ", opt_age)
# st.write("You ethnicity: ", enc_ethnicity(opt_eth))
# st.write("You sensitivity on price: ", opt_sensOnPrice)
# st.write("You sensitivity on calorie: ", opt_sensOnK)
# st.write("You sensitivity on food: ", opt_sensSeasonalFood)
# st.write("Hot or Cold: ", opt_HorK)
# st.write("Allergy: ", enc_allergy(opt_allergy))

# st.write("\n")

if btnSubmit:
    user_F_M = list()
    user_F_M.append(enc_gender(opt_gender))
    user_F_M.append(opt_age)
    user_F_M.append(enc_ethnicity(opt_eth))
    user_F_M.append(opt_sensOnPrice)
    user_F_M.append(opt_sensOnK)
    user_F_M.append(opt_sensSeasonalFood)
    user_F_M.append(enc_horc(opt_HorK))
    user_F_M += enc_allergy(opt_allergy)


    # st.write("User feature vector: ", user_F_M)

    st.write("Our recommendation: ", get_recommend(user_F_M))


# image_listing = list()
# for item in retrieve_all_image(IMAGE_DIR):
#     image_listing.append(Image.open(IMAGE_DIR+"/"+item))

# st.image(image_listing)
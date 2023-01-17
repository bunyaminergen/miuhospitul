import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_methods import predict

classes = {0:'Diabetes',1:'No Diabetes'}

class_labels = list(classes.values())

st.title("Diabetes Predict")
st.markdown('**Objective** : Given details about your contidion and we will show you ... ')
st.markdown('The model can predict if it belongs to the following three Categories : **Diabetes , not diabetes** ')

def predict_class():

    data = list(map(float, [AGE,
                            SEX,
                            HIGHCHOL,
                            CHOLCHECK,
                            BMI,
                            SMOKER,
                            HEARTDISEASEORATTACK,
                            PHYSACTIVITY,
                            FRUITS,
                            VEGGIES,
                            HVYALCOHOLCONSUMP,
                            GENHLTH,
                            MENTHLTH,
                            PHYSHLTH,
                            DIFFWALK,
                            STROKE,
                            HIGHBP,
                            NEW_MENT_GEN,
                            NEW_MENT_PHY,
                            NEW_MENT_DIFF,
                            NEW_PHY_GEN,
                            NEW_PHY_DIFF,
                            NEW_BMI_HIGHBP,
                            NEW_BMI_GENHLTH,
                            NEW_BMI_DIFF,
                            NEW_AGE_HEARTDIS,
                            NEW_AGE_HIGHCHOL,
                            NEW_AGE_HIGHBP,
                            NEW_AGE_DIFF,
                            NEW_HEARTDIS_AGE,
                            NEW_HEARTDIS_GEN,
                            NEW_HEARTDIS_DIFF,
                            NEW_HEARTDIS_STROKE,
                            NEW_HEARTDIS_HIGHBP,
                            NEW_HIGHBP_DIFF,
                            NEW_HIGHBP_HIGHCHOL,
                            NEW_HIGHBP_GENHLTH,
                            NEW_HIGHBP_STROKE,
                            NEW_STROKE_GEN,
                            NEW_GEN_DIFF,
                            NEW_FRUITS_VEGGIES,
                            NEW_DIFF_MENTHLTH,
                            NEW_DIFF_AGE,
                            NEW_BMI_AGE,
                            NEW_BMI_HIGHCHOL,
                            NEW_MENTHLTH_AGE,
                            NEW_SMOKER_HEARTDIS,
                            NEW_SEX_HEARTDIS,
                            NEW_HEALTH,
                            NEW_HEALTH_STROKE,
                            NEW_HEALTH_DIFFWALK,
                            NEW_PHYSACT_HEARTATTACK,
                            NEW_PHYSACT_HIGHCHOL,
                            NEW_PHYSACT_STROKE]))

    result, probs = predict(data)

    st.write("The predicted class is ",result)

    probs = [np.round(x,2) for x in probs]

    print(probs)

    ax = sns.barplot(x = probs,y =class_labels, palette="winter", orient='h')

    ax.set_yticklabels(class_labels,rotation=0)

    plt.title("Probabilities of the Data belonging to each class")

    for index, value in enumerate(probs):
        plt.text(value, index,str(value))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

st.markdown("**Please enter the details of health**")

AGE                  = float(st.number_input('Enter AGE'))
SEX                  = float(st.number_input('Enter SEX'))
HIGHCHOL             = float(st.number_input('Enter HIGHCHOL'))
CHOLCHECK            = float(st.number_input('Enter CHOLCHECK'))
BMI                  = float(st.number_input('Enter BMI'))
SMOKER               = float(st.number_input('Enter SMOKER'))
HEARTDISEASEORATTACK = float(st.number_input('Enter HEARTDISEASEORATTACK'))
PHYSACTIVITY         = float(st.number_input('Enter PHYSACTIVITY'))
FRUITS               = float(st.number_input('Enter FRUITS'))
VEGGIES              = float(st.number_input('Enter VEGGIES'))
HVYALCOHOLCONSUMP    = float(st.number_input('HVYALCOHOLCONSUMP'))
GENHLTH              = float(st.number_input('Enter GENHLTH'))
MENTHLTH             = float(st.number_input('Enter MENTHLTH'))
PHYSHLTH             = float(st.number_input('Enter PHYSHLTH'))
DIFFWALK             = float(st.number_input('Enter DIFFWALK'))
STROKE               = float(st.number_input('Enter STROKE'))
HIGHBP               = float(st.number_input('Enter HIGHBP'))

NEW_MENT_GEN            = MENTHLTH * GENHLTH
NEW_MENT_PHY            = MENTHLTH * PHYSHLTH
NEW_MENT_DIFF           = MENTHLTH * DIFFWALK
NEW_PHY_GEN             = PHYSHLTH * GENHLTH
NEW_PHY_DIFF            = PHYSHLTH * DIFFWALK

NEW_BMI_HIGHBP          = BMI * HIGHBP
NEW_BMI_GENHLTH         = BMI * GENHLTH
NEW_BMI_DIFF            = BMI * DIFFWALK

NEW_AGE_HEARTDIS        = AGE * HEARTDISEASEORATTACK
NEW_AGE_HIGHCHOL        = AGE * HIGHCHOL
NEW_AGE_HIGHBP          = AGE * HIGHBP
NEW_AGE_DIFF            = AGE * DIFFWALK

NEW_HEARTDIS_AGE        = HEARTDISEASEORATTACK + AGE
NEW_HEARTDIS_GEN        = HEARTDISEASEORATTACK + GENHLTH
NEW_HEARTDIS_DIFF       = HEARTDISEASEORATTACK + DIFFWALK
NEW_HEARTDIS_STROKE     = HEARTDISEASEORATTACK + STROKE
NEW_HEARTDIS_HIGHBP     = HEARTDISEASEORATTACK + HIGHBP

NEW_HIGHBP_DIFF         = HIGHBP + DIFFWALK
NEW_HIGHBP_HIGHCHOL     = HIGHBP + HIGHCHOL
NEW_HIGHBP_GENHLTH      = HIGHBP + GENHLTH
NEW_HIGHBP_STROKE       = HIGHBP + STROKE

NEW_STROKE_GEN          = STROKE + GENHLTH
NEW_GEN_DIFF            = GENHLTH + DIFFWALK
NEW_FRUITS_VEGGIES      = FRUITS + VEGGIES
NEW_AGE_HEARTDIS        = AGE + HEARTDISEASEORATTACK

NEW_DIFF_MENTHLTH       = DIFFWALK + MENTHLTH
NEW_DIFF_AGE            = DIFFWALK + AGE

NEW_BMI_AGE             = BMI + AGE
NEW_BMI_HIGHCHOL        = BMI + HIGHCHOL

NEW_MENTHLTH_AGE        = MENTHLTH + AGE
NEW_SMOKER_HEARTDIS     = SMOKER + HEARTDISEASEORATTACK
NEW_SEX_HEARTDIS        = SEX + HEARTDISEASEORATTACK

NEW_HEALTH              = GENHLTH + MENTHLTH + PHYSHLTH
NEW_HEALTH_STROKE       = NEW_HEALTH + STROKE
NEW_HEALTH_DIFFWALK     = NEW_HEALTH + DIFFWALK

NEW_PHYSACT_HEARTATTACK = PHYSACTIVITY + HEARTDISEASEORATTACK
NEW_PHYSACT_HIGHCHOL    = PHYSACTIVITY + HIGHCHOL
NEW_PHYSACT_STROKE      = PHYSACTIVITY + STROKE

if st.button("Predict"):
    predict_class()

"""
# (MENTHLTH*GENHLTH), variable3, variable4]))

AGE = [1 if 17<int(st.text_input('Age', ''))<25 else 2]
c = 20

d = [1 if 17<c<25 else 2 if 25<c<29 else 3 if 29<c<30]

d

# df["Bmi_Cat"] = [1 if i < 18.5 else 2 if 18.5<i<24.9 else 3 if 24.9<|i|<29.9 else 4 if 29.9<i<34.9 else 5 for i in df["Body_mass_index"]]

[int(st.text_input('Enter Mental Health', '')) if ]

# AgeGroups    Counts    AgeRange
#    1.0         979      18-24
#    2.0        1396      25-29
#    3.0        2049      30-34
#    4.0        2793      35-39
#    5.0        3520      40-44
#    6.0        4648      45-49
#    7.0        6872      50-54
#    8.0        8603      55-59
#    9.0       10112      60-64
#   10.0       10856      65-69
#   11.0        8044      70-74
#   12.0        5394      75-70
#   13.0        5426        80+

"""

# pip3 freeze > requirements.txt  # Python3
# pip freeze > requirements.txt  # Python2
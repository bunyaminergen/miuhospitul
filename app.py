########################################################################################################################
########################################################################################################################
# Miuhospitul
########################################################################################################################
########################################################################################################################

import streamlit as st

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

def predict(arr):
    # Load the model
    with open('final_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'Diyabetsiniz!',1:'Diyabet değilsiniz!'}

    # return prediction as well as class probabilities

    preds = model.predict_proba([arr])[0]

    return (classes[np.argmax(preds)], preds)

classes = {0:'Diyabetsiniz!',1:'Diyabet değilsiniz!'}

class_labels = list(classes.values())

st.title("Miuhospitul Hastahanesi")

st.title("Diyabet Tahmin Yapay Zeka Uygulamasına Hoş Geldiniz !")

st.markdown("**Yapay Zeka** : Lütfen diyabet olup olmadığınızı öğrenmek için aşağıdaki bilgileri doldurunuz ! ")
st.markdown("Bilgileri doldurduktan sonra Enter'a basınız. Sonuçlar : **Diyabetsiniz** veya **Diyabet değilsiniz** şeklinde görünecektir !")

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

    st.write("**Sonuç:**",result)

    probs = [np.round(x,2) for x in probs]

    ax = sns.barplot(x = probs,y =class_labels, palette="winter", orient='h')

    ax.set_yticklabels(class_labels, rotation=0)

    plt.title("Diyabet olma ihtimaliniz !")

    for index, value in enumerate(probs):
        plt.text(value, index,str(value))

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

st.markdown("**Lütfen bilgileri doldurunuz**")

Input_Age = float(st.number_input('Yaşınız'))
AGE                  = float((1 if 17<Input_Age<25 else 2 if 24<Input_Age<30 else 3 if 29<Input_Age<35 else 4 if 34<Input_Age<40 else 5 if 39<Input_Age<45 else 6 if 44<Input_Age<50 else 7 if 49<Input_Age<55 else 8 if 54<Input_Age<60 else 9 if 59<Input_Age<65 else 10 if 64<Input_Age<70 else 11 if 69<Input_Age<75 else 12 if 74<Input_Age<80 else 13 if Input_Age>79 else 0))

SEX                  = float(st.number_input("Cinsiyetiniz (**Erkek = 1** , **Kadın = 0**)"))
HIGHCHOL             = float(st.number_input("Yüksek Kolesterol (**Varsa = 1** , **Yoksa = 0**)"))
CHOLCHECK            = float(st.number_input("Son 5 senede kolesterol kontrolü yaptırdınız mı ? (**Evet = 1** , **Hayır = 0**)"))

BOY                  = float(st.number_input('Boyunuz nedir ? ', value = 1))
KILO                 = float(st.number_input('Kilonuz nedir ? ', value = 1))
BMI                  = (KILO / (BOY**2))

SMOKER               = float(st.number_input('Sigara içiyor musunuz ? (**Evet = 1** , **Hayır = 0**)'))
HEARTDISEASEORATTACK = float(st.number_input("Koroner Kalp Hastalığız var mı ? (**Evet = 1** , **Hayır = 0**)"))
PHYSACTIVITY         = float(st.number_input('Son 30 günde fiziksel aktivite yaptınız mı ? (**Evet = 1** , **Hayır = 0**)'))
FRUITS               = float(st.number_input('Günde en az 1 meyve tüketiyor musunuz ? (**Evet = 1** , **Hayır = 0**)'))
VEGGIES              = float(st.number_input('Günde en az 1 sebze tüketiyor musunuz ? (**Evet = 1** , **Hayır = 0**)'))
HVYALCOHOLCONSUMP    = float(st.number_input("Haftada 10 bardaktan fazla alkol tüketiyor musunuz ? (**Evet = 1** , **Hayır = 0**)"))
GENHLTH              = float(st.number_input("Genel sağlık durumunuz nedir? (**Muhteşem = 1** , **Çok iyi = 2**, **İyi = 3**, **İyi değil = 4**, **Kötü = 5**)"))
MENTHLTH             = float(st.number_input("Son 30 günde kendinizi psikolojik olarak kötü hissetiğiniz gün sayısı nedir ? "))
PHYSHLTH             = float(st.number_input("Son 30 günde fiziksel sakatlık ve/veya yaralanma geçirdiğiniz gün sayısı nedir ? " ))
DIFFWALK             = float(st.number_input("Yürümekte ya da merdiven çıkmakta zorlanıyor musunuz ? (**Evet = 1** , **Hayır = 0**)"))
STROKE               = float(st.number_input("Hiç inme geçirdiniz mi ? (**Evet = 1** , **Hayır = 0**)"))
HIGHBP               = float(st.number_input("Yüksek Tansiyonunuz var mı ? (**Evet = 1** , **Hayır = 0**)"))

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

if st.button("Enter"):
    predict_class()

# pip3 freeze > requirements.txt  # Python3
# pip freeze > requirements.txt  # Python2
# streamlit run app.py


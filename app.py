import streamlit as st
import pandas as pd
import pypickle
from sklearn import preprocessing

loaded_model = pypickle.load('model.pkl')


def prediction(data):
    df = pd.DataFrame(data)

    label = preprocessing.LabelEncoder()

    lab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i in lab:
        df.iloc[i] = label.fit_transform(df.iloc[i])
    num_data = df.values.reshape(1, -1)

    pred = loaded_model.predict(num_data)

    if pred[0] == 0:
        return "Patient do not have diabetes"
    else:
        return "Patient have diabetes"


def main():
    st.title("Diabetes Predictive Model")
    age = st.number_input("Age range of the individuals : 1-20 to 65")
    gender = st.text_input("Gender information : Male, Female")
    polyuria = st.text_input("Presence of excessive urination : Yes, No")
    polydipsia = st.text_input("Excessive thirst : Yes, No")
    sudden_weight_loss = st.text_input("Abrupt weight loss : Yes, No")
    weakness = st.text_input("Generalized weakness : Yes, No")
    polyphagia = st.text_input("Excessive hunger : Yes, No")
    genital_thrush = st.text_input("Presence of genital thrush : Yes, No")
    visual_blurring = st.text_input("Blurring of vision : Yes, No")
    itching = st.text_input("Presence of itching : Yes, No")
    irritability = st.text_input("Display of irritability : Yes, No")
    delayed_healing = st.text_input("Delayed wound healing : Yes, No")
    partial_paresis = st.text_input("Partial loss of voluntary movement : Yes, No")
    muscle_stiffness = st.text_input("Presence of muscle stiffness : Yes, No")
    alopecia = st.text_input("Hair loss : Yes, No")
    obesity = st.text_input("Presence of obesity : Yes, No")

    Diabetes = ""

    if st.button("Result"):
        Diabetes = prediction([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
                               polyphagia, genital_thrush, visual_blurring, itching,
                               irritability, delayed_healing, partial_paresis, muscle_stiffness,
                               alopecia, obesity])

    st.success(Diabetes)


if __name__ == "__main__":
    main()

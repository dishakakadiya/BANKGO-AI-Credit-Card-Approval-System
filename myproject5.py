import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

# === Sidebar ===
with st.sidebar:
    selected = option_menu(
        "Admin Panel",
        ["Dashboard", "About us", "Dataset", "Prediction Form", "Data Visualization", "Login", "Settings"],
        icons=["cast", "people", "table", "check-square", "bar-chart", "lock", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )

# === Load data ===
df = pd.read_csv("CD.csv")

# === Preprocess data for model ===
df.fillna(0, inplace=True)

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === Train Random Forest ===
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# === Dashboard ===
if selected == "Dashboard":
    st.header("Welcome to the Dashboard")
    st.subheader("Project: Credit Card Approval System")

# === About Us ===s
elif selected == "About us":
    st.header("About Us")
    st.subheader("BANKGO: AI Credit Card Approval")
    st.write("""
    AI credit card approval uses machine learning to assess credit card applications 
    faster and more accurately than traditional manual reviews.
    """)
    st.subheader("Key Features")
    st.write("- Age, Gender, Marital Status")
    st.write("- Annual Income, Employment Type")
    st.write("- Credit Score, Default History")
    st.write("- Number of Dependents, Housing Type")

    st.subheader("Future Enhancements")
    st.write("- Add credit limit prediction")
    st.write("- Integrate with credit bureaus for real-time scores")
    st.write("- Implement fraud detection modules")

# === Dataset ===
elif selected == "Dataset":
    st.header("Project Dataset")
    df = pd.read_csv("CD.csv")
    st.dataframe(df)

# === Prediction Form ===
elif selected == "Prediction Form":
    st.header("Credit Card Approval Form")

    sample_row = df.iloc[0]

    st.write("Fill the form to check prediction. Below is pre-filled with sample data:")

    with st.form("prediction_form"):
        # Removed Ind_ID — not used for prediction
        Ind_ID = st.number_input("Ind_ID", min_value=0, value=int(sample_row["Ind_ID"]))
        GENDER = st.selectbox("Gender", ["M", "F"], index=0 if sample_row["GENDER"] == 1 else 1)
        Car_Owner = st.selectbox("Car Owner", ["Y", "N"], index=0 if sample_row["Car_Owner"] == 1 else 1)
        Propert_Owner = st.selectbox("Property Owner", ["Y", "N"], index=0 if sample_row["Propert_Owner"] == 1 else 1)

        CHILDREN = st.number_input("Children", min_value=0, value=int(sample_row["CHILDREN"]))
        Annual_income = st.number_input("Annual Income", min_value=0, value=int(sample_row["Annual_income"]))
        Type_Income = st.selectbox("Type Income", list(label_encoders['Type_Income'].classes_))
        Type_Occupation = st.selectbox("Type Occupation", list(label_encoders['Type_Occupation'].classes_))
        EDUCATION = st.selectbox("Education", list(label_encoders['EDUCATION'].classes_))
        Marital_status = st.selectbox("Marital Status", list(label_encoders['Marital_status'].classes_))
        Housing_type = st.selectbox("Housing Type", list(label_encoders['Housing_type'].classes_))
        Birthday_count = st.number_input("Birthday Count", value=int(sample_row["Birthday_count"]))
        Employed_days = st.number_input("Employed Days", value=int(sample_row["Employed_days"]))
        Mobile_phone = st.selectbox("Mobile Phone", [0, 1], index=int(sample_row["Mobile_phone"]))
        Work_Phone = st.selectbox("Work Phone", [0, 1], index=int(sample_row["Work_Phone"]))
        Phone = st.selectbox("Phone", [0, 1], index=int(sample_row["Phone"]))
        EMAIL_ID = st.selectbox("EMAIL_ID", [0, 1], index=int(sample_row["EMAIL_ID"]))
        Family_Members = st.number_input("Family Members", min_value=1, max_value=20, value=int(sample_row["Family_Members"]))

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = {
                "Ind_ID": Ind_ID,
                "GENDER": label_encoders['GENDER'].transform([GENDER])[0],
                "Car_Owner": label_encoders['Car_Owner'].transform([Car_Owner])[0],
                "Propert_Owner": label_encoders['Propert_Owner'].transform([Propert_Owner])[0],
                "CHILDREN": CHILDREN,
                "Annual_income": Annual_income,
                "Type_Income": label_encoders['Type_Income'].transform([Type_Income])[0],
                "Type_Occupation": label_encoders['Type_Occupation'].transform([Type_Occupation])[0],
                "EDUCATION": label_encoders['EDUCATION'].transform([EDUCATION])[0],
                "Marital_status": label_encoders['Marital_status'].transform([Marital_status])[0],
                "Housing_type": label_encoders['Housing_type'].transform([Housing_type])[0],
                "Birthday_count": Birthday_count,
                "Employed_days": Employed_days,
                "Mobile_phone": Mobile_phone,
                "Work_Phone": Work_Phone,
                "Phone": Phone,
                "EMAIL_ID": EMAIL_ID,
                "Family_Members": Family_Members
            }

            input_df = pd.DataFrame([input_data])

            # Make sure column order matches training data
            input_df = input_df[X.columns]

            # Predict
            prediction = rf_model.predict(input_df)[0]
            if prediction == 1:
                st.success("✅ The credit card is likely to be APPROVED.")
            else:
                st.error("❌ The credit card is likely to be REJECTED.")

if selected == "Data Visualization":
    st.header("Data Visualization")
    st.title("CREDIT CARD APPROVAL DATA VISUALIZATION")

    # Correct data
    creditcard = [
        "Ind_ID", "GENDER", "Car_Owner", "Property_Owner", "CHILDREN",
        "Annual_income", "Type_Income", "EDUCATION", "Marital_status",
        "Housing_type", "Birthdate_count", "Employed_days", "Mobile_phone",
        "Work_Phone", "Phone", "EMAIL_ID", "Employed_Staff", "Family_Members", "Label"
    ]

    share = [10, 20, 30, 15, 15, 10, 10, 15, 11, 8, 12, 15, 10, 5, 5, 10, 15, 10, 15]  # 19 values

    # Get a Seaborn color palette with enough colors
    colors = sns.color_palette('pastel', len(creditcard))

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10,20))  # Adjust size here!

    # Plot pie chart
    ax.pie(
        share,
        labels=creditcard,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )

    ax.set_title("CREDIT CARD APPROVAL")

    # Display in Streamlit
    st.pyplot(fig)

# === Login ===
elif selected == "Login":
    st.header("Login Page")
    with st.form(key="login_form"):
        email = st.text_input("Enter your ID")
        password = st.text_input("Enter your Password", type="password")
        submit = st.form_submit_button("Submit")
        if submit:
            if email == "abc@gmail.com" and password == "1234":
                st.success("Login successful")
            else:
                st.error("Invalid details")

# === Settings ===
elif selected == "Settings":
    st.info("Settings page is under development")

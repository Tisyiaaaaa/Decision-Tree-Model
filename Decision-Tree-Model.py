import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training data and predict data
data = pd.read_csv('training-data.csv')
test_data = pd.read_csv("predict-data.csv")

# Filtering the training data to only filter when the 'Type' == 'HHI'
data_hhi = data[data['Type'] == 'HHI']


# This function is to remove symbols and replace with whitespace
# Define a function to preprocess the 'sec' column
def preprocess_sec(sec_value):
    # Remove symbols from the 'sec' value and replace them with whitespace
    sec_value = ''.join(char if char.isalnum() else ' ' for char in sec_value)
    return sec_value.strip()

# # Apply the preprocessing function to the 'sec' column and fill null values in the 'target_column'
data_hhi['pageTitle'] = data_hhi.apply(lambda row: row['sec'] if pd.isnull(row['pageTitle']) else row['pageTitle'], axis=1)

# Preprocess the 'sec' column using the defined function
data_hhi['pageTitle'] = data_hhi['pageTitle'].apply(preprocess_sec)

# Replace null values in the specified column with 'replacement_value'
replacement_value = '(not set)'
data_hhi['mobileDeviceBranding'] = data_hhi['mobileDeviceBranding'].fillna(replacement_value)
data_hhi['mobileDeviceModel'] = data_hhi['mobileDeviceModel'].fillna(replacement_value)
data_hhi['mobileInputSelector'] = data_hhi['mobileInputSelector'].fillna(replacement_value)
data_hhi['mobileDeviceMarketingName'] = data_hhi['mobileDeviceMarketingName'].fillna(replacement_value)


data_hhi['sub_sec'] = data_hhi['sub_sec'].fillna('none')



# Define the feature that will be used in both training and predict data

training_data = data_hhi.loc[:, ['y', 'onPageTime', 'nextTime', 'user_active_days', 'total_visit_counts'
                                    ,'total_unique_session', 'total_time_spent', 'time_spent_per_session', 
                                    'timeOnPage', 'continent', 'subContinent', 'country']]



predict_data = test_data.loc[:, ['onPageTime', 'nextTime', 'user_active_days', 'total_visit_counts'
                                    ,'total_unique_session', 'total_time_spent', 'time_spent_per_session', 
                                    'timeOnPage', 'continent', 'subContinent', 'country']]


# Label Encoding for continent
label_encoder_cont = LabelEncoder()

# Combine the unique colors from both data 
cont_unique = pd.concat([training_data["continent"], predict_data["continent"]]).unique()
cont_enc_fit = label_encoder_cont.fit(cont_unique)

training_data["Continent_Encoded"] = label_encoder_cont.transform(training_data['continent'])
predict_data["Continent_Encoded"] = label_encoder_cont.transform(predict_data['continent'])


# Label Encoding for sub continent
label_encoder_sub_cont = LabelEncoder()

# Combine the unique colors from both data frames
sub_cont_unique = pd.concat([training_data["subContinent"], predict_data["subContinent"]]).unique()
sub_cont_enc_fit = label_encoder_sub_cont.fit(sub_cont_unique)

training_data["SubContinent_Encoded"] = label_encoder_sub_cont.transform(training_data['subContinent'])
predict_data["SubContinent_Encoded"] = label_encoder_sub_cont.transform(predict_data['subContinent'])

# Label Encoding for country
label_encoder_country = LabelEncoder()

# Combine the unique colors from both data frames
country_unique = pd.concat([training_data["country"], predict_data["country"]]).unique()
country_enc_fit = label_encoder_country.fit(country_unique)

training_data["Country_Encoded"] = label_encoder_country.transform(training_data['country'])
predict_data["Country_Encoded"] = label_encoder_country.transform(predict_data['country'])


# Label Encoding for income 'y'
label_encoder_income = LabelEncoder()

# Combine the unique colors from both data frames
income = training_data['y']
conun_enc_fit = label_encoder_income.fit(income)

training_data["Income_Encoded"] = label_encoder_income.transform(training_data['y'])


# Make a copy of the training data and predict data to ensure no data loss
training_df = training_data.copy()
predicting_df = predict_data.copy()

# Dropping the original column that was used for encoding in order to run the model
training_df.drop("continent", axis=1, inplace=True)
training_df.drop("subContinent", axis=1, inplace=True)
training_df.drop("country", axis=1, inplace=True)
training_df.drop("y", axis=1, inplace=True)

predicting_df.drop("continent", axis=1, inplace=True)
predicting_df.drop("subContinent", axis=1, inplace=True)
predicting_df.drop("country", axis=1, inplace=True)


# Filling in mean value for null data in time_per_session and total_time_spent
mean_value_session = predicting_df['time_spent_per_session'].mean()
predicting_df['time_spent_per_session'].fillna(mean_value_session, inplace=True)

mean_value_spent = predicting_df['total_time_spent'].mean()
predicting_df['total_time_spent'].fillna(mean_value_spent, inplace=True)


########## Decision Tree Model Starts here #################

# Split the dataset into features (X) and the encoded income bins (y_encoded)
X = training_df.drop(['Income_Encoded'], axis=1)
y_encoded = training_df['Income_Encoded']

# Split the data into training and testing sets
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.8, random_state=42, stratify=y_encoded)

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier()

# Train the model
decision_tree.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred_encoded = decision_tree.predict(X_test)


# To change back the encoded value into its original form
y_pred_original = label_encoder_income.inverse_transform(y_encoded)

# Compare the predicted income bins with the original income bins
y_test_original = training_df.loc[y_test_encoded.index, 'Income_Encoded']

# Create a new column to store predicted income bins alongside original income bins
training_data['predicted_income'] = label_encoder_income.inverse_transform(y_encoded)


# Save train model in .pkl
joblib.dump(decision_tree, 'model_path.pkl')

# Load the model
loaded_model = joblib.load('model_path.pkl')

# Prediction using the prediction data using the model saved
new_predictions = loaded_model.predict(predicting_df)

# Adding the prediction in a new column
predicting_df['predicted_target'] = new_predictions


# Change back the encoded value of the prediction value into its original form and putting it in the original predict data 
test_data['Predict_Income'] = label_encoder_income.inverse_transform(predicting_df['predicted_target'])

test_data.to_csv('prediction_data_output.csv', index=False)

print('-- SUCCESFULLY RUN --')
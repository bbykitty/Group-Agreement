import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''
# Sample data: Replace this with your actual dataset
data = {
    'text': [
        'This is a great product',
        'I am very satisfied with the service',
        'The quality is not good',
        'Highly recommended',
        'Very disappointed with the purchase'
    ],
    'estimation': [5, 4, 2, 5, 1]  # Numerical estimations
}

# Convert to DataFrame
df = pd.DataFrame(data)
'''
df = pd.read_csv("all_groups_kripp.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Agreement'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
'''
new_text = ["It broke as I took it out of the packaging"]
new_text_tfidf = vectorizer.transform(new_text)
prediction = model.predict(new_text_tfidf)
print(f'Predicted estimation: {prediction[0]}')
'''
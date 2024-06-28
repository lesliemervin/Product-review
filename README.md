# Product-review
Determining the reviews given for a product (Reference: Amazon product review dataset)

#DataPreprocessing 
#TrainingTheModel
#EvaluatingTheModel

import pandas as pd 
df = pd.read_csv('amazon_product.csv')
df.head(4)
df.info()
data = df[["reviews.text", "reviews.rating"]]
data = data.rename(columns={"reviews.text": "reviews", "reviews.rating": "ratings"})
import re
def clean(text):
    cleaned_text = text.lower()
    cleaned_text = re.sub('[^A-Za-z0-9]+'," ",cleaned_text)
    cleaned_text = re.sub('[0-9]+'," ",cleaned_text)
    return cleaned_text
clean("what a great-product ! everyone should $buy this")
clean("100% bad")
data ["cleaned_reviews"] = data["reviews"].apply(clean)
data.sample(4)
from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer()
X_encoded = vectoriser.fit_transform(data["cleaned_reviews"]).toarray()
X_encoded
vectoriser.vocabulary_
vectoriser.get_feature_names_out()
Y = data["ratings"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded,Y, test_size=0.2, shuffle=True)

# USING NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100
X_encoded = vectoriser.transform([X_unknown_text]).toarray()
model.predict(X_encoded)

# USING DECISION TREE CLASSIFIER 

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,y_pred)*100
confusion_matrix(y_test,y_pred)


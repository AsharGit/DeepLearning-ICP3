from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Embedding, Flatten

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values


#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
# Mistake 1: texts_to_matrix is not ideal when dealing with thousands of words.
sentences = tokenizer.texts_to_sequences(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# Mistake 2:input_dim was undefined.
input_dim = 2000
print(input_dim)

model = Sequential()
model.add(Embedding(input_dim, 50, input_length=500))
model.add(Flatten())
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
# Mistake 3: sigmoid activation is used for output between 0 and 1
# This does not work for our target values.
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

DATA_PATH = './data/'
GLOVE_EMBEDDING_PATH = '../glove.6B.300d.txt'
SEED = 42
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 10
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

np.random.seed(SEED)
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)


def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def build_model():

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(
        vocab_size,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)

    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Conv1D(256, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-4))(x)

    outputs = Dense(1, activation='sigmoid')(x)

    # Define the model
    model = tf.keras.Model(sequence_input, outputs)
    # Print model summary
    model.summary()

    return model


def train_model(model):
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['acc']
    )

    # Create a save directory
    checkpoint_dir = os.path.join(os.getcwd(), 'model/')

    # Define callbacks
    # Save models with highest 'val_acc'
    checkpoint_cb = ModelCheckpoint(
        checkpoint_dir + 'model.{epoch:02d}-{val_acc:.2f}.h5',
        save_best_only=True,
    )
    # Stop training if 'val_acc' decreases for 5 times
    early_stopping_cb = EarlyStopping(monitor='val_acc', patience=5)
    # Save model log
    csv_logger_cb = CSVLogger(checkpoint_dir + 'training.log')
    # Define LR optimizer
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    # Define a learning rate schedule
    # lr_schedule = ExponentialDecay(LR, decay_steps=100000, decay_rate=0.96, staircase=True)

    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),  # validation_split=0.1,
        callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb, reduce_lr_cb]
    )

    return history


if __name__ == '__main__':
    # Reading JSON file
    df = pd.read_json(DATA_PATH + 'wos2class.json')
    # Convert column names to lowercase, columns: 'title', 'abstract', 'label'
    df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
    # Select 'abstract' as a feature
    # df = df.drop(['title'], axis=1)
    # Create a feature column
    df['feature'] = df[['title', 'abstract']].apply(lambda x: ''.join(x), axis=1)
    # Preprocessing feature column
    df.feature = df.feature.apply(lambda x: preprocess(x))
    # Split dataset into training and testing set
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED)

    print("Dataset size:", len(df))
    print("Train set size:", len(train_df))
    print("Test set size", len(test_df))

    # Save train and test sets as new json files
    train_df.to_json(r'./data/wos2class.train.json')
    test_df.to_json(r'./data/wos2class.test.json')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df.feature)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size :", vocab_size)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df.feature),
                            maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df.feature),
                           maxlen=MAX_SEQUENCE_LENGTH)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Encode the classification label as Material Science: 1, Chemistry: 0
    le = LabelEncoder()
    le.fit(train_df.label)
    y_train = le.transform(train_df.label)
    y_test = le.transform(test_df.label)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    embeddings_index = {}
    f = open(GLOVE_EMBEDDING_PATH)
    for line in f:
        values = line.split()
        word = value = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    model = build_model()
    history = train_model(model)

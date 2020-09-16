import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("gnn.log"),
                              logging.StreamHandler()])

# Read Data
df = pd.read_table('./diabetes.txt',header=None,encoding='gb2312',sep='\t')
df.astype(float)
# remove redundant col which is the opposite value of the 10th col
df.pop(10)
# remove first col of bias = 1
df.pop(0)
# the label column
label = df.pop(9)

# train feature
train_feature = df[:576]
# train label
train_label = label[:576]
# test feature
test_feature = df[576:]
# test label
test_label = label[576:]

model = Sequential([
    Dense(6, input_shape=(8,), activation='sigmoid', bias_initializer='ones', kernel_initializer='random_uniform'),
    Dense(6, activation='sigmoid'),
    Dense(4, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
              loss='mean_squared_error',
              metrics=['accuracy'])

epochs = 1000
history = model.fit(train_feature.values, train_label.values, epochs=epochs)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.ylabel('accuracy')
plt.xlabel('epoch')

test_loss, test_acc = model.evaluate(test_feature,  test_label, verbose=2)
print('\nTest accuracy:', test_acc)

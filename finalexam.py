import tensorflow as tf
import numpy as np
import os

# 1. 確保 model 資料夾存在
os.makedirs('model', exist_ok=True)

# 2. 載入並預處理資料
# 處理方式必須與 model_test.py 完全一致
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"訓練資料形狀: {x_train.shape}")
print(f"測試資料形狀: {x_test.shape}")

# 3. 建立一個符合專案要求的模型架構
# 只使用 nn_predict.py 支援的 Flatten 和 Dense(relu/softmax) 層
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(512, activation='relu', name='dense_2'),

    tf.keras.layers.Dense(256, activation='relu', name='dense_3'),
    tf.keras.layers.Dense(256, activation='relu', name='dense_4'),
    tf.keras.layers.Dense(128, activation='relu', name='dense_5'),
    tf.keras.layers.Dense(64, activation='relu', name='dense_6'),

    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

model.summary()

# 4. 編譯模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 訓練模型
print("\n開始訓練模型...")
model.fit(
    x_train, y_train,
    batch_size=1024,
    epochs=100, # 你可以調整此數值以獲得更好的準確率
    validation_data=(x_test, y_test),
    verbose=1
)

# 6. 提取並保存權重為 .npz 檔案
print("\n保存模型權重...")
weights_dict = {}
for layer in model.layers:
    # 使用 isinstance 檢查是最穩健的方法
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()
        if weights:
            weights_dict[f'{layer.name}/kernel:0'] = weights[0]
            weights_dict[f'{layer.name}/bias:0'] = weights[1]
            print(f"正在保存層: {layer.name}")

np.savez('model/fashion_mnist.npz', **weights_dict)
print("\n模型權重已成功保存到 model/fashion_mnist.npz")

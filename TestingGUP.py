import tensorflow as tf
import os
import pickle
from IPython.display import clear_output
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto(allow_soft_placement = True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
# config.gpu_options.allow_growth = True
 
# sess0 = tf.InteractiveSession(config = config)

# def mutual_info_loss(c, c_given_x):
#     """The mutual information metric we aim to minimize"""
#     eps = 1e-8
#     conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
#     entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

#     return conditional_entropy + entropy
'''
生成器（generator）
首先，建立一個“生成器（generator）”模型，它將一個向量（從潛在空間 - 在訓練期間隨機取樣）轉換為候選影象。
GAN通常出現的許多問題之一是generator卡在生成的影象上，看起來像噪聲。一種可能的解決方案是在鑑別器（discriminator）
和生成器（generator）上使用dropout。
'''
import keras
from keras import layers
import numpy as np

num_classes = 52
latent_dim = 46
height = 92
width = 68
channels = 3

# 兩個input，分別是亂數、類別
latent = keras.Input(shape=(latent_dim,))
image_class = Input(shape=(1,), dtype='int32')
cls = Embedding(num_classes, latent_dim,
                embeddings_initializer='glorot_normal')(image_class)

generator_input = layers.add([latent, cls])
# 首先，將輸入轉換為16x16 128通道的feature map
# x = layers.Dense(32 * 46 * 17)(generator_input)
# x = layers.LeakyReLU()(x)
# # x = layers.Dense(128 * 46 * 17)(x)
# # x = layers.LeakyReLU()(x)
x = layers.Dense(128 * 46 * 34)(generator_input)
x = layers.LeakyReLU()(x)
#x = layers.Dropout(0.3)(x)
x = layers.Reshape((46, 34, 128))(x)

# 然後，添加捲積層
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)

# 上取樣至 32 x 32
x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.Dropout(0.2)(x)
x = layers.LeakyReLU()(x)
# 新增更多的卷積層
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)

# 生成一個 32x32 1-channel 的feature map
x = layers.Conv2D(channels, 16, activation='tanh', padding='same')(x)
generator = keras.models.Model([latent, image_class], x)
generator.summary()


# In[8]:


'''
discriminator(鑑別器)
建立鑑別器模型，它將候選影象（真實的或合成的）作為輸入，並將其分為兩類：“生成的影象”或“來自訓練集的真實影象”。
'''
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(128)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(64, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
#x = layers.Dropout(0.25)(x)
x = layers.Conv2D(32, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Dense(32)(x)
x = layers.LeakyReLU()(x)
#x = layers.Dropout(0.25)(x)
x = layers.Conv2D(32, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
# 重要的技巧（新增一個dropout層）
x = layers.Dropout(0,4)(x)

# 分類層
isFake = layers.Dense(1, activation='sigmoid')(x)
classDetac = layers.Dense(num_classes, activation ='softmax')(x)

discriminator = keras.models.Model(discriminator_input, [isFake,classDetac])
discriminator.summary()

# In[11]:
# generator = pickle.load(open('generatorClass.p','rb'))
# discriminator = pickle.load(open('discriminatorClass.p','rb'))
# 為了訓練穩定，在優化器中使用學習率衰減和梯度限幅（按值）。
discriminator_optimizer = keras.optimizers.adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, amsgrad=False,epsilon=1e-08)
discriminator.compile(optimizer=discriminator_optimizer, loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])


# In[16]:


'''
The adversarial network:對抗網路
最後，設定GAN，它連結生成器（generator）和鑑別器（discrimitor）。 這是一種模型，經過訓練，
將使生成器（generator）朝著提高其愚弄鑑別器（discrimitor）能力的方向移動。 該模型將潛在的空間點轉換為分類決策，
“假的”或“真實的”，並且意味著使用始終是“這些是真實影象”的標籤來訓練。 所以訓練`gan`將以一種方式更新
“發生器”的權重，使得“鑑別器”在檢視假影象時更可能預測“真實”。 非常重要的是，將鑑別器設定為在訓練
期間被凍結（不可訓練）：訓練“gan”時其權重不會更新。 如果在此過程中可以更新鑑別器權重，那麼將訓練鑑別
器始終預測“真實”。
'''
# 將鑑別器（discrimitor）權重設定為不可訓練（僅適用於`gan`模型）

discriminator.trainable = False

fakeImg = generator([latent, image_class])
fake, aux = discriminator(fakeImg)

gan = keras.models.Model([latent, image_class], [fake, aux])
#losses = [ mutual_info_loss]
gan_optimizer = keras.optimizers.adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, amsgrad=False,epsilon=1e-08)
gan.compile(optimizer=gan_optimizer, loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
gan.summary()

# In[19]:


# gan = pickle.load(open('gan.p','rb'))
'''
  開始訓練了。
  每個epoch：
   *在潛在空間中繪製隨機點（隨機噪聲）。
   *使用此隨機噪聲生成帶有“generator”的影象。
   *將生成的影象與實際影象混合。
   *使用這些混合影象訓練“鑑別器”，使用相應的目標，“真實”（對於真實影象）或“假”（對於生成的影象）。
   *在潛在空間中繪製新的隨機點。
   *使用這些隨機向量訓練“gan”，目標都是“這些是真實的影象”。 這將更新發生器的權重（僅因為鑑別器在“gan”內被凍結）
   以使它們朝向獲得鑑別器以預測所生成影象的“這些是真實影象”，即這訓練發生器欺騙鑑別器。
 '''
import os
from keras.preprocessing import image
import cv2
import numpy as np
'''
# 匯入CIFAR10資料集
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

# 從CIFAR10資料集中選擇frog類（class 6）
x_train = x_train[y_train.flatten() == 6]

# 標準化資料
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
'''

dirPath = 'C:/Users/User/GAN/SA'
SA = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
x_train = []
for img in SA:
    imgcv = cv2.imread(dirPath+'/'+img)
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    x_train.append(imgcv)
x_train = np.array(x_train)
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

lableImg = pickle.load(open('lableImg.p','rb'))

iterations = 450000000000
batch_size = 40
save_dir = '.\\gan_image3'

start = 0 
# 開始訓練迭代
for step in range(15200,iterations):
    # 在潛在空間中抽樣隨機點
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 隨機選擇化哪個類別
    random_labels = np.random.randint(0, num_classes, batch_size)
    # 將隨機抽樣點解碼為假影象
    generated_images = generator.predict([random_latent_vectors,random_labels.reshape((-1, 1))])
    
    # 將假影象與真實影象進行比較
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])
    
    # 取真實類別的標籤
    
    
    # 組裝區別真假影象的標籤
    labels = np.concatenate([np.ones((batch_size, 1)),
                            np.zeros((batch_size, 1))])
    # 組裝區別真假類別的標籤
    claLabels = lableImg[start: stop]
    combined_clsLable = np.concatenate([random_labels, claLabels])
    #combined_clsLable = np.concatenate([np.negative(np.ones((batch_size))), claLabels])
    
    # 重要的技巧，在標籤上新增隨機噪聲
    labels += 0.05 * np.random.random(labels.shape)
    
    # Assign sample weight
    disc_sample_weight = [np.ones(2 * (batch_size)),
                                  np.concatenate((np.zeros(batch_size) ,
                                                  np.ones(batch_size)*3))]
    # 訓練鑑別器（discrimitor）
    d_loss = discriminator.train_on_batch(combined_images, [labels,combined_clsLable], sample_weight=disc_sample_weight)
    

    
    # 彙集標有“所有真實影象”的標籤
    misleading_targets = np.zeros((batch_size, 1))
    
    # 再做一個新的亂數和標籤
    # 在潛在空間中抽樣隨機點
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 隨機選擇化哪個類別
    random_labels = np.random.randint(0, num_classes, batch_size)
    
    # 訓練生成器（generator）（通過gan模型，鑑別器（discrimitor）權值被凍結）
    a_loss = gan.train_on_batch([random_latent_vectors,random_labels.reshape((-1, 1))], [misleading_targets,random_labels])
    
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        # 儲存網路權值
        gan.save_weights('gan.h5')
        if step % 100 == 0:
            print(discriminator.predict(combined_images))
        # 輸出metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # 儲存生成的影象
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, str(step) + '.png'))
        
        pickle.dump(discriminator,open('discriminatorClass.p','wb'))
        pickle.dump(generator,open('generatorClass.p','wb'))
        pickle.dump(gan,open('ganClass.p','wb'))
        
        # 儲存真實影象，以便進行比較
#         img = image.array_to_img(real_images[0] * 255., scale=False)
#         img.save(os.path.join(save_dir, 'real_SA' + str(step) + '.png'))
    if step % 1000 == 0:
        clear_output()

# 繪圖
import matplotlib.pyplot as plt

# 在潛在空間中抽樣隨機點
random_latent_vectors = np.random.normal(size=(10, latent_dim))

# 將隨機抽樣點解碼為假影象
generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()
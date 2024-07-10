import tensorflow as tf
import numpy as np

class TemporalAnimeTransformer:
    def __init__(self, frame_size=(256, 256, 3), sequence_length=16):
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.style_embedding_dim = 256
        self.content_embedding_dim = 512
        
        self.temporal_encoder = self.build_temporal_encoder()
        self.style_encoder = self.build_style_encoder()
        self.content_encoder = self.build_content_encoder()
        self.anime_generator = self.build_anime_generator()
        self.temporal_discriminator = self.build_temporal_discriminator()
        self.vgg = self.build_vgg()
        
    def build_temporal_encoder(self):
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, *self.frame_size))
        x = tf.keras.layers.ConvLSTM2D(64, 3, padding='same', return_sequences=True)(inputs)
        x = tf.keras.layers.ConvLSTM2D(128, 3, padding='same', return_sequences=True)(x)
        x = tf.keras.layers.ConvLSTM2D(256, 3, padding='same', return_sequences=False)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        outputs = tf.keras.layers.Dense(self.content_embedding_dim)(x)
        return tf.keras.Model(inputs, outputs)
    
    def build_style_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.frame_size)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(self.style_embedding_dim)(x)
        return tf.keras.Model(inputs, outputs)
    
    def build_content_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.frame_size)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(x)
        outputs = tf.keras.layers.LeakyReLU(0.2)(x)
        return tf.keras.Model(inputs, outputs)
    
    def build_anime_generator(self):
        content_input = tf.keras.layers.Input(shape=(None, None, 256))
        style_input = tf.keras.layers.Input(shape=(self.style_embedding_dim,))
        temporal_input = tf.keras.layers.Input(shape=(self.content_embedding_dim,))
        
        x = self.adain(content_input, style_input)
        x = self.self_attention(x)
        
        for _ in range(6):
            x = self.style_fusion_block(x, style_input)
        
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(256, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        
        x = tf.keras.layers.UpSampling2D()(x)
        outputs = tf.keras.layers.Conv2D(3, 7, padding='same', activation='tanh')(x)
        
        return tf.keras.Model([content_input, style_input, temporal_input], outputs)
    
    def adain(self, content, style):
        mean, var = tf.nn.moments(content, axes=[1, 2], keepdims=True)
        normalized = (content - mean) / tf.sqrt(var + 1e-8)
        
        style_mean = tf.keras.layers.Dense(content.shape[-1])(style)
        style_var = tf.keras.layers.Dense(content.shape[-1])(style)
        
        style_mean = tf.reshape(style_mean, [-1, 1, 1, content.shape[-1]])
        style_var = tf.reshape(style_var, [-1, 1, 1, content.shape[-1]])
        
        return normalized * tf.sqrt(style_var + 1e-8) + style_mean
    
    def self_attention(self, x):
        batch_size, height, width, channels = x.shape
        
        f = tf.keras.layers.Conv2D(channels // 8, 1)(x)
        g = tf.keras.layers.Conv2D(channels // 8, 1)(x)
        h = tf.keras.layers.Conv2D(channels, 1)(x)
        
        s = tf.matmul(tf.reshape(g, [batch_size, -1, height * width]), 
                      tf.reshape(f, [batch_size, height * width, -1]), transpose_b=True)
        
        beta = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(beta, tf.reshape(h, [batch_size, -1, channels]))
        o = tf.reshape(o, [batch_size, height, width, channels])
        
        return tf.keras.layers.Add()([x, o])
    
    def style_fusion_block(self, x, style):
        style_attention = tf.keras.layers.Dense(x.shape[-1])(style)
        style_attention = tf.keras.layers.Reshape((1, 1, -1))(style_attention)
        style_attention = tf.keras.layers.multiply([x, style_attention])
        
        residual = x
        x = tf.keras.layers.Conv2D(x.shape[-1], 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(x.shape[-1], 3, padding='same')(x)
        x = tf.keras.layers.Add()([x, residual, style_attention])
        return tf.keras.layers.LeakyReLU(0.2)(x)
    
    def build_temporal_discriminator(self):
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, *self.frame_size))
        
        def discriminator_block(x, filters, strides=1):
            x = tf.keras.layers.Conv3D(filters, (3, 3, 3), strides=(1, strides, strides), padding='same')(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            return x
        
        x = discriminator_block(inputs, 64, 2)
        x = discriminator_block(x, 128, 2)
        x = discriminator_block(x, 256, 2)
        x = discriminator_block(x, 512)
        
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def build_vgg(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        return tf.keras.Model(inputs=vgg.inputs, outputs=vgg.get_layer('block4_conv1').output)
    
    @tf.function
    def train_step(self, real_sequence, anime_style_image):
        with tf.GradientTape(persistent=True) as tape:
            temporal_features = self.temporal_encoder(real_sequence)
            content_features = self.content_encoder(real_sequence[:, -1])
            style_features = self.style_encoder(anime_style_image)
            
            generated_frame = self.anime_generator([content_features, style_features, temporal_features])
            
            real_validity = self.temporal_discriminator(real_sequence)
            fake_sequence = tf.concat([real_sequence[:, :-1], tf.expand_dims(generated_frame, 1)], axis=1)
            fake_validity = self.temporal_discriminator(fake_sequence)
            
            d_loss = self.discriminator_loss(real_validity, fake_validity)
            g_loss = self.generator_loss(fake_validity)
            content_loss = tf.reduce_mean(tf.abs(content_features - self.content_encoder(generated_frame)))
            style_loss = tf.reduce_mean(tf.abs(style_features - self.style_encoder(generated_frame)))
            temporal_consistency_loss = tf.reduce_mean(tf.abs(real_sequence[:, -1] - generated_frame))
            perceptual_loss = self.perceptual_loss(real_sequence[:, -1], generated_frame)
            
            total_g_loss = g_loss + content_loss + style_loss + temporal_consistency_loss + perceptual_loss
        
        d_gradients = tape.gradient(d_loss, self.temporal_discriminator.trainable_variables)
        g_gradients = tape.gradient(total_g_loss, self.anime_generator.trainable_variables +
                                    self.temporal_encoder.trainable_variables +
                                    self.content_encoder.trainable_variables +
                                    self.style_encoder.trainable_variables)
        
        self.d_optimizer.apply_gradients(zip(d_gradients, self.temporal_discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.anime_generator.trainable_variables +
                                             self.temporal_encoder.trainable_variables +
                                             self.content_encoder.trainable_variables +
                                             self.style_encoder.trainable_variables))
        
        return d_loss, total_g_loss

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def perceptual_loss(self, real_image, generated_image):
        real_features = self.vgg(real_image)
        generated_features = self.vgg(generated_image)
        return tf.reduce_mean(tf.abs(real_features - generated_features))

    def transform_video(self, video_frames, anime_style_image):
        anime_frames = []
        for i in range(0, len(video_frames) - self.sequence_length + 1):
            sequence = video_frames[i:i+self.sequence_length]
            sequence = np.array(sequence) / 127.5 - 1  # Normalize to [-1, 1]
            sequence = np.expand_dims(sequence, axis=0)
            
            anime_style_image = np.array(anime_style_image) / 127.5 - 1
            anime_style_image = np.expand_dims(anime_style_image, axis=0)
            
            temporal_features = self.temporal_encoder(sequence)
            content_features = self.content_encoder(sequence[:, -1])
            style_features = self.style_encoder(anime_style_image)
            
            generated_frame = self.anime_generator([content_features, style_features, temporal_features])
            generated_frame = (generated_frame.numpy() + 1) * 127.5  # Denormalize
            generated_frame = np.clip(generated_frame, 0, 255).astype(np.uint8)
            
            anime_frames.append(generated_frame[0])
        
        return anime_frames
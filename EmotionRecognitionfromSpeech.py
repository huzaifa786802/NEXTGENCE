import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, TimeDistributed, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Configuration
DATASET_PATH = "ravdess/audio_speech_actors_01-24/"
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_FRAMES = int(SAMPLE_RATE * DURATION / HOP_LENGTH)

# Feature extraction function
def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or truncate to fixed length
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, int(SAMPLE_RATE * DURATION - len(audio))), mode='constant')
        else:
            audio = audio[:int(SAMPLE_RATE * DURATION)]
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Ensure consistent shape
        if mfccs.shape[1] < N_FRAMES:
            mfccs = np.pad(mfccs, ((0, 0), (0, N_FRAMES - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :N_FRAMES]
        
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Load dataset
def load_dataset():
    features = []
    labels = []
    
    for actor_folder in tqdm(os.listdir(DATASET_PATH), desc="Processing actors"):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if not os.path.isdir(actor_path):
            continue
            
        for file_name in os.listdir(actor_path):
            if not file_name.endswith('.wav'):
                continue
                
            # Parse emotion from filename
            parts = file_name.split('-')
            emotion_code = parts[2]
            emotion = EMOTIONS.get(emotion_code)
            
            if emotion is None:
                continue
                
            file_path = os.path.join(actor_path, file_name)
            mfccs = extract_features(file_path)
            
            if mfccs is not None:
                features.append(mfccs)
                labels.append(emotion)
    
    return np.array(features), np.array(labels)

# Load and preprocess data
print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build model
model = Sequential([
    # CNN layers
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(N_MFCC, N_FRAMES, 1)),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    
    # LSTM layers
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    
    # Dense layers
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(EMOTIONS), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save('emotion_recognition_model.h5')
print("Model saved as 'emotion_recognition_model.h5'")
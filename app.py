import os
import torch
import numpy as np
import librosa
import scipy.signal
import pickle
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model components
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('scaler_rnn.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('maxlen.pkl', 'rb') as f:
    max_len = pickle.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN_GRU(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.gru = torch.nn.GRU(input_dim, 320, num_layers=1, bidirectional=False, batch_first=True, dropout=0.0)
        self.attention = torch.nn.MultiheadAttention(embed_dim=320, num_heads=8, dropout=0.25, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(320, 384),
            self.activation,
            torch.nn.LayerNorm(384),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(384, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        pooled = attn_out.mean(dim=1)
        return self.classifier(pooled)

model = RNN_GRU(input_dim=40, num_classes=len(label_encoder.classes_)).to(device)
model.load_state_dict(torch.load('testing_rnn.pt', map_location=device))
model.eval()

# Feature extraction functions (same as your PNCC pipeline)
def extract_pncc(data, sr, n_mels=40, n_fft=512, hop_length=160, win_length=400):
    try:
        data, _ = librosa.effects.trim(data, top_db=40)
        emphasized = scipy.signal.lfilter([1, -0.97], [1], data)
        stft = np.abs(librosa.stft(emphasized, n_fft=n_fft, hop_length=hop_length, win_length=win_length)) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_spec = np.dot(mel_basis, stft).T
        padded = np.pad(mel_spec, ((2, 2), (0, 0)), mode='constant')
        mt_power = np.stack([np.mean(padded[i:i + 5], axis=0) for i in range(mel_spec.shape[0])])

        def asym_lowpass(x, alpha=0.999, beta=0.5):
            y = np.zeros_like(x)
            y[0] = x[0]
            for i in range(1, len(x)):
                y[i] = np.where(x[i] >= y[i - 1], alpha * y[i - 1] + (1 - alpha) * x[i], beta * y[i - 1] + (1 - beta) * x[i])
            return y

        def temporal_masking(x, lam=0.85, mu=0.2):
            peak = np.zeros_like(x)
            masked = np.zeros_like(x)
            peak[0] = x[0]
            masked[0] = x[0]
            for i in range(1, len(x)):
                peak[i] = np.maximum(lam * peak[i - 1], x[i])
                masked[i] = np.where(x[i] >= lam * peak[i - 1], x[i], mu * peak[i - 1])
            return masked

        def weight_smoothing(x, ref, N=4):
            smoothed = np.zeros_like(x)
            for t in range(x.shape[0]):
                for f in range(x.shape[1]):
                    l1 = max(f - N, 0)
                    l2 = min(f + N + 1, x.shape[1])
                    smoothed[t, f] = np.mean(x[t, l1:l2] / (ref[t, l1:l2] + 1e-6))
            return smoothed

        low_env = asym_lowpass(mt_power)
        subtracted = mt_power - low_env
        rectified = np.maximum(0, subtracted)
        floor = asym_lowpass(rectified)
        masked = temporal_masking(rectified)
        switched = np.where(mt_power >= 2 * low_env, masked, floor)
        smoothed = weight_smoothing(switched, mt_power)
        tf_norm = mel_spec * smoothed
        mean_power = np.zeros(tf_norm.shape[0])
        mean_power[0] = 0.0001
        for i in range(1, tf_norm.shape[0]):
            mean_power[i] = 0.999 * mean_power[i - 1] + 0.001 * np.mean(tf_norm[i])
        norm = tf_norm / (mean_power[:, None] + 1e-6)
        nonlin = norm ** (1 / 15)
        return (nonlin - np.mean(nonlin)) / (np.std(nonlin) + 1e-6)
    except:
        return None

def pad_or_truncate(arr, max_len):
    if arr.shape[0] < max_len:
        pad_width = max_len - arr.shape[0]
        return np.pad(arr, ((0, pad_width), (0, 0)), mode='constant')
    else:
        return arr[:max_len]

def predict_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=40)
    spec = extract_pncc(y, sr)
    if spec is None:
        return "PNCC extraction failed", 0
    spec = pad_or_truncate(spec, max_len)
    scaled = scaler.transform(spec)
    x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        return label_encoder.classes_[pred_idx], probs[pred_idx] * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audiofile' not in request.files:
            return redirect(request.url)
        file = request.files['audiofile']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            pred, confidence = predict_from_audio(filepath)
            os.remove(filepath)
            return render_template('index.html', prediction=pred, confidence=confidence)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

class LandmarkSequenceDataset(Dataset):
    def __init__(self, root, split, seq_len=16, stride=2):
        meta = json.load(open(Path(root)/"meta.json"))
        # meta["clips"] looks like {"forward":[[file1,...], ...], ...}
        self.samples = []
        for cls, clips in meta[split].items():
            for clip in clips:
                arr = np.load(clip)            # (T, 63)
                # slice into windows
                for i in range(0, len(arr)-seq_len*stride, stride):
                    window = arr[i : i+seq_len*stride : stride]
                    self.samples.append((window.astype(np.float32),
                                         meta["class_to_idx"][cls]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.tensor(y)

class GRUGesture(nn.Module):
    def __init__(self, input_dim=63, hidden=128, num_layers=2, num_classes=21):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, num_classes))
    def forward(self, x):          # x: (B, L, 63)
        out, _ = self.rnn(x)
        feat = out[:, -1]          # last time-step
        return self.classifier(feat)

class TemporalConvGesture(nn.Module):
    def __init__(self, input_dim=63, num_classes=21, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hid, 3, padding=1, groups=1),
            nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 3, padding=1, groups=hid), # depthwise
            nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 3, padding=1),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(hid, num_classes)
    def forward(self, x):          # (B, L, 63)
        x = x.transpose(1,2)       # -> (B, 63, L)
        feat = self.net(x).squeeze(-1)
        return self.fc(feat)

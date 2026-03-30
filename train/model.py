import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# x → MelSpectrogram → dB → TDNN1~5 → Stats Pooling → fc → embedding → ArcFace

#  --- AAM-Softmax (ArcFace) ---
# 取代fc8的全連接層，改成ArcMarginProduct，讓模型在訓練時學習到更有區分度的特徵空間
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        """
        in_features: 輸入的 X-vector 維度 (512)
        out_features: 輸出的類別數 (總語者人數 2682) 
        s: 縮放因子 (Scale)，強制放大特徵空間，通常設 30
        m: 角度邊界 (Margin)，通常設 0.2
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s # scale 30
        self.m = m # margin 0.2 rad
        
        # 權重矩陣，大小為 [num_classes, 特徵維度 512]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化權重，讓訓練更穩定

        # 預先計算 cos(m) 和 sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # 處理邊界的常數 (當 theta + m 超過 180 度沒繼續遞減)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. 將特徵 X 和權重 W 都做 L2 normalization (長度變成 1)
        # |W||X|cos(theta) -> cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 如果沒有傳入 label (在評估測試時)，直接回傳縮放後的相似度
        if label is None:
            return cosine * self.s
            
        # 2. 計算 cos(theta + m)
        # 三角函數展開: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7)) # sin^2 + cos^2 = 1 算出 sin(theta)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # easy_margin : https://blog.csdn.net/qq_29168809/article/details/123247675
        # 邊界保護：如果原本的角度已經大於 180-m 度，就不套用公式，改用泰勒展開近似
        # theta + m 超過 180 度後，cosine 會變成負值，這時候就不繼續遞減了
        # theta + m > 180 -> cosine < cos(180 - m) = th
        # where (if cosine > th) use phi, else use cosine - mm
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 3. 建立 One-hot 矩陣 (只在正確答案的位置加上 margin，其他位置維持原本的 cosine)
        # one_hot 是一個和 cosine 同形狀的矩陣，只有在正確類別的位置是 1，其他位置是 0
        # cosine shape: [Batch, num_classes]
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1) # 在 label 指定的位置放 1，其他位置是 0
        
        # 4. 融合結果
        # 在正確類別的位置使用 phi (cos(theta + m))，其他位置使用原本的 cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 5. 乘上縮放因子 s
        output *= self.s
        
        return output

# --- 加入 SE Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        channels: 輸入的特徵通道數 (例如 512)
        reduction: 降維比例，用來減少參數運算量 (512 -> 32 -> 512)
        """
        super(SEBlock, self).__init__()
        # Squeeze: 在時間軸 (dim=2) 上做全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: 降維 -> ReLU -> 升維 -> Sigmoid 產生 0~1 的權重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time]
        b, c, _ = x.size()
        
        # 1. Squeeze: 壓扁時間軸 -> [Batch, Channels, 1] -> [Batch, Channels]
        y = self.avg_pool(x).view(b, c)
        
        # 2. Excitation: 算出每個 Channel 的重要性權重 -> [Batch, Channels]
        y = self.fc(y)
        
        # 3. 變回 [Batch, Channels, 1] 以便跟原來的特徵相乘
        y = y.view(b, c, 1)
        
        # 4. 加權輸出
        return x * y.expand_as(x)

# --- TDNN ---
class TDNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation=1):
        super(TDNNLayer, self).__init__()
        # TDNN 的卷積層，使用 dilation 來擴大視野
        self.conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=context_size, 
            dilation=dilation
        )
        # batch normalization 來正則化卷積輸出到單位長度
        self.bn = nn.BatchNorm1d(output_dim)
        # ReLU 激活函數來增加非線性
        self.relu = nn.ReLU()

        # 在卷積層之後，直接裝上 SE Block
        self.se = SEBlock(channels=output_dim)

    # 前向傳播：卷積 -> BN -> ReLU
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 讓 SE Block 對特徵通道進行動態篩選
        x = self.se(x) 
        return x

# --- ASP ---
class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        """
        in_dim: 輸入的特徵通道數 (TDNN5 輸出是 1500)
        bottleneck_dim: Attention 網路的瓶頸層維度，用來減少參數與計算量
        """
        super(AttentiveStatisticsPooling, self).__init__()
        # 建立一個小型的 Attention 網路來計算每個 Frame 的分數
        # 使用 kernel_size=1 的 Conv1d 相當於對每個時間點做全連接層 (Linear)
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)
        )

    def forward(self, x):
        # x 的形狀: [Batch, Channels, Time] ([Batch, 1500, 301])
        
        # 1. 計算 Attention 原始分數 (Logits)
        # alpha_logits 的形狀: [Batch, Channels, Time]
        alpha_logits = self.attention(x) 
        
        # 2. 套用 Softmax 轉成權重分布
        # 對時間維度 (dim=2) 做 Softmax，確保每個 Channel 在所有時間點的權重總和為 1
        alphas = F.softmax(alpha_logits, dim=2)
        
        # 3. 計算加權平均 (Weighted Mean)
        # 把原始特徵 x 乘上權重 alphas，然後在時間維度上加總
        # mean 的形狀會變成: [Batch, Channels]
        mean = torch.sum(alphas * x, dim=2)
        
        # 4. 計算加權標準差 (Weighted Standard Deviation)
        # 公式: sqrt( sum(alpha * x^2) - mean^2 )
        # torch.clamp 加上 1e-6 是為了數值穩定性，防止 sqrt(0) 產生 NaN 導致模型崩潰
        var = torch.sum(alphas * (x ** 2), dim=2) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        
        # 5. 拼接特徵
        # 將 mean 和 std 拼接起來，輸出形狀: [Batch, Channels * 2] (3000 維)
        return torch.cat((mean, std), dim=1)

# --- XVector model ---
class XVector(nn.Module):
    def __init__(self, num_classes, input_dim=80): # 變成 80 維特徵
        super(XVector, self).__init__()
        
        # 1. 特徵提取 (Mel Spectrogram + DB)
        # 和dataset相同參數
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            n_mels=input_dim, # 80
            hop_length=160 # 10ms
        )
        # AmplitudeToDB 把能量轉成分貝
        self.db_transform = T.AmplitudeToDB()
        
        # 2. TDNN Layers 
        # L_out ​= L_in ​− dilation×(context_size − 1)
        self.tdnn1 = TDNNLayer(input_dim, 512, context_size=5, dilation=1) # [Batch, 512, Time]
        self.tdnn2 = TDNNLayer(512, 512, context_size=3, dilation=2)
        self.tdnn3 = TDNNLayer(512, 512, context_size=3, dilation=3)
        self.tdnn4 = TDNNLayer(512, 512, context_size=1, dilation=1)    
        self.tdnn5 = TDNNLayer(512, 1500, context_size=1, dilation=1)

        # 加入 ASP 模組，輸入維度對齊 TDNN5 的輸出 1500
        self.asp = AttentiveStatisticsPooling(in_dim=1500)

        # 3. Statistical Pooling 和全連接層
        # self.fc6 = nn.Linear(3000, 512)
        # self.bn6 = nn.BatchNorm1d(512)
        
        # self.fc7 = nn.Linear(512, 512)
        # self.bn7 = nn.BatchNorm1d(512)

        # 3. 唯一的 取代原本的 fc6 與 fc7
        # ==========================================
        self.fc = nn.Linear(3000, 512)
        self.bn = nn.BatchNorm1d(512)
        
        # 4. 最後的分類層 (改成 ArcMarginProduct)
        self.arcface = ArcMarginProduct(in_features=512, out_features=num_classes)

    def forward(self, x, label=None):
        # 輸入 x 的 shape: [Batch, time] 
        
        # A. 特徵提取 
        # x: [Batch, 48000] -> [Batch, 80, 301]
        x = self.mel_transform(x)
        x = self.db_transform(x)
        
        # B. TDNN
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)

        # C. Attentive Statistics Pooling (取代原本的 Mean + Std)
        # x 輸入形狀: [Batch, 1500, Time]
        # x 輸出形狀: [Batch, 3000]
        x = self.asp(x)

        
        # 512 維的 x-vector embedding
        # x-vector ，shape: [Batch, 512] 
        
        # D. Embedding 
        # ==========================================
        # 直接將 3000 維濃縮成 512 維的 x-vector embedding
        embedding = self.bn(self.fc(x))

        # 訓練時：fc7套用 ReLU，強迫模型在極小空間內練出抗噪能力
        if self.training:
            embedding = F.relu(embedding)
        # 評估時：不套用 ReLU，讓特徵分布更自然，提升 Cosine Similarity 的表現

        # E. Classification (arcface)
        logits = self.arcface(embedding, label)
        
        return logits, embedding
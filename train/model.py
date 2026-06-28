import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

"""
定義語者特徵萃取的深度學習模型。
資料流為 : 原始Waveform -> Mel-Spectrogram 
-> TDNN 捕捉特徵 -> SE Block 篩選重要通道 
-> ASP 將變動長度的時序特徵濃縮為固定長度
-> 線性層降維產生 X-Vector -> 送入 ArcFace 計算分類 Logits
"""
# ==========================================
# 1. ArcFace (AAM-Softmax)
# ==========================================
class ArcMarginProduct(nn.Module):
    """
    ArcFace 分類層。
    有別於傳統的 Softmax，ArcFace 透過在特徵空間中加入角度邊界 (Margin)，
    強迫同一個語者的特徵更集中，不同語者的特徵推得更遠，強化語者辨識力。
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # s (Scale): 縮放因子，將 Cosine 值拉大，使 Softmax 能產生陡峭的機率分布
        self.s = s 
        # m (Margin): 角度邊界，強迫正確類別的預測角度比其他類別更嚴苛
        self.m = m 
        
        # 權重矩陣：代表每一個語者在特徵空間中的中心
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) 

        # 預先計算三角函數常數，減少 Forward 運算量
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        # 邊界保護常數 (Easy Margin)：
        # 當特徵角度 (theta) + margin 超過 180 度時，cos 會失去單調遞減的特性。
        # 這些常數用於在極端角度下改用泰勒展開近似，維持梯度的穩定性。
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. L2 正規化：將輸入特徵與權重都投影到單位球面上
        # 計算結果即為 Cosine 相似度
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 測試/推論階段：沒有 Label 時，直接回傳縮放後的 Cosine 相似度
        if label is None:
            return cosine * self.s
            
        # 2. 加入 Margin
        # 根據公式 cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m) 展開
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 若原本角度已超過 180-m 度，使用近似法代替公式
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 3. 套用 One-hot 遮罩
        # 只在「正確答案」的對應位置套用加上 Margin 的 phi，
        # 其餘錯誤類別的位置維持原本的 cosine。
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 4. 縮放並輸出
        return output * self.s


# ==========================================
# 2. TDNN + SE + ASP 
# ==========================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (通道注意力機制)。
    用於動態評估每個特徵通道的重要性，並給予對應的權重。
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 壓縮時間軸，取得每個通道的全域資訊
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: 透過降維再升維的結構，學習通道間的非線性關係
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # 將學習到的權重 (0~1) 乘回原始特徵
        return x * y.expand_as(x)


class TDNNLayer(nn.Module):
    """
    Time-Delay Neural Network 層。
    利用 1D 卷積與擴張 (Dilation) 在時間軸上捕捉上下文特徵。
    此版本在末端整合了 SEBlock 來強化特徵表達。
    """
    def __init__(self, input_dim, output_dim, context_size, dilation=1):
        super(TDNNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=context_size, 
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.se = SEBlock(channels=output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x) 
        return x


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP)。
    傳統 X-Vector 使用全域平均與標準差。ASP 引入 Attention 機制，
    讓模型學會「把注意力放在有特徵明顯的幀上」，忽略靜音或噪音干擾。
    """
    def __init__(self, in_dim, bottleneck_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        # 使用 1D 卷積實作時間的全連接層
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)
        )

    def forward(self, x):
        # 1. 產生每個時間步的權重 (Alpha)
        alpha_logits = self.attention(x) 
        alphas = F.softmax(alpha_logits, dim=2)
        
        # 2. 計算加權平均 (Weighted Mean)
        mean = torch.sum(alphas * x, dim=2)
        
        # 3. 計算加權標準差 (Weighted Standard Deviation)
        # 此處的 torch.clamp(..., min=1e-6) 絕對不可移除，
        # 若變異數為負或 0，計算平方根會產生 NaN，導致整個模型梯度毀損。
        var = torch.sum(alphas * (x ** 2), dim=2) - (mean ** 2)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        
        # 將 Mean 與 Std 拼接，維度翻倍 (1500 -> 3000)
        return torch.cat((mean, std), dim=1)


# ==========================================
# 3. X-Vector 主架構
# ==========================================
class XVector(nn.Module):
    """
    增強版 X-Vector 網路架構。
    包含：Mel-Spectrogram 前處理 -> 5層 TDNN -> ASP 池化 -> 線性投影層 -> ArcFace 分類。
    """
    def __init__(self, num_classes, input_dim=80): 
        super(XVector, self).__init__()
        
        # A. 前處理特徵提取層
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            n_mels=input_dim, # 80 維 Mel 頻譜特徵
            win_length=400,   # 25 ms
            hop_length=160,   # 10 ms
        )
        self.db_transform = T.AmplitudeToDB()

        # B. 特徵提取 (Frame-level Feature Extraction)
        # 透過堆疊 TDNN 擴大感受野 (Receptive Field)
        self.tdnn1 = TDNNLayer(input_dim, 512, context_size=5, dilation=1) 
        self.tdnn2 = TDNNLayer(512, 512, context_size=3, dilation=2)
        self.tdnn3 = TDNNLayer(512, 512, context_size=3, dilation=3)
        self.tdnn4 = TDNNLayer(512, 512, context_size=1, dilation=1)
        # MFA: tdnn5 改接收 x2 + x3 + x4 的拼接特徵
        self.tdnn5 = TDNNLayer(1536, 1500, context_size=1, dilation=1)

        # C. 池化層 (Pooling)
        self.asp = AttentiveStatisticsPooling(in_dim=1500)

        # D. 特徵投影 (Segment-level Embedding)
        # 精簡了雙層 FC，改用單層線性映射直接輸出 512 維特徵
        self.fc = nn.Linear(3000, 512)
        self.bn = nn.BatchNorm1d(512)
        
        # E. 損失分類
        self.arcface = ArcMarginProduct(in_features=512, out_features=num_classes)

    def forward(self, x, label=None):
        # 維度變化：
        # 初始輸入 x: [Batch, Time_Samples] (e.g. [256, 48000])
        
        # A. 特徵提取 
        x = self.mel_transform(x)
        x = self.db_transform(x)
        # 轉換後 x: [Batch, 80, Frames] (e.g. [256, 80, 301])

        # B. TDNN 卷積特徵
        x1 = self.tdnn1(x)  # -> [Batch, 512, Frames']
        x2 = self.tdnn2(x1)
        x3 = self.tdnn3(x2)
        x4 = self.tdnn4(x3)
        # MFA: 將中間層特徵融合後再送入 tdnn5
        # tdnn2 的時間長度比 tdnn3/tdnn4 長，先裁到與 x4 相同
        t = x4.size(2)
        x2_crop = x2[:, :, :t] # 裁剪 tdnn2 的時間維度到與 tdnn4 相同
        x3_crop = x3  # x3 與 x4 等長
        x4_crop = x4
        # MFA: 將 tdnn2、tdnn3、tdnn4 的特徵在通道維度拼接，形成 512*3=1536 維的特徵
        x_mfa = torch.cat((x2_crop, x3_crop, x4_crop), dim=1)  # -> [Batch, 1536, Frames'']
        x5 = self.tdnn5(x_mfa)  # -> [Batch, 1500, Frames'']

        # C. 統計池化：將變動長度的時間軸壓扁 (mean + std)
        x = self.asp(x5)    # -> [Batch, 3000]

        # D. X-Vector 嵌入層
        # 這裡不加 ReLU 激活函數。
        # 讓特徵分布在完整的空間 
        embedding = self.bn(self.fc(x)) # -> [Batch, 512]

        # E. ArcFace 分類 (輸出 Logits)
        logits = self.arcface(embedding, label) # -> [Batch, num_classes]
        
        # 回傳分類分數與特徵向量，供訓練與 EER 評估使用
        return logits, embedding
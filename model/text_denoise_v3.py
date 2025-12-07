import torch
import torch.nn as nn
import torch.nn.functional as F

class TextDenoisingModule(nn.Module):
    def __init__(self, word_dim, sent_dim):
        super(TextDenoisingModule, self).__init__()
        
        # 单词级别注意力层
        self.word_attention = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.Tanh(),
            nn.Linear(word_dim, 1)
        )
        
        # 句子级别注意力层
        self.sent_attention = nn.Sequential(
            nn.Linear(sent_dim, sent_dim),
            nn.Tanh(),
            nn.Linear(sent_dim, 1)
        )
        
        # 门控机制，动态决定是否保留信息
        self.word_gate = nn.Sigmoid()
        self.sent_gate = nn.Sigmoid()

    def forward(self, txt_sent, txt_word):
        # txt_sent: [b, 512]
        # txt_word: [b, 77, 512]
        
        # 1. 对每个单词的特征进行加权（单词级别的注意力）
        word_scores = self.word_attention(txt_word)  # [b, 77, 1]
        word_scores = word_scores.squeeze(-1)  # [b, 77]
        word_attention_weights = F.softmax(word_scores, dim=-1)  # [b, 77]
        
        # 用注意力权重对单词特征进行加权平均
        txt_word_denoised = torch.bmm(word_attention_weights.unsqueeze(1), txt_word)  # [b, 1, 512]
        txt_word_denoised = txt_word_denoised.squeeze(1)  # [b, 512]
        
        # 2. 动态门控：决定是否保留每个单词的特征
        word_gate_weights = self.word_gate(txt_word_denoised)  # [b, 512]
        word_gate_weights = F.softmax(word_gate_weights, dim=-1)  # 归一化

        # 对每个单词特征应用门控（升维到 [b, 77, 512]）
        word_gate_weights_expanded = word_gate_weights.unsqueeze(1).expand(-1, txt_word.size(1), -1)  # [b, 77, 512]
        txt_word_denoised = txt_word * word_gate_weights_expanded  # [b, 77, 512]

        # 3. 对句子特征进行加权（句子级别的注意力）
        sent_scores = self.sent_attention(txt_sent)  # [b, 1]
        sent_scores = sent_scores.squeeze(-1)  # [b]
        sent_attention_weights = F.softmax(sent_scores, dim=-1)  # [b]
        
        # 用注意力权重对句子特征进行加权
        txt_sent_denoised = txt_sent * sent_attention_weights.unsqueeze(-1)  # [b, 512]

        # 4. 动态门控：决定是否保留句子特征
        sent_gate_weights = self.sent_gate(txt_sent_denoised)  # [b, 512]
        sent_gate_weights = F.softmax(sent_gate_weights, dim=-1)  # 归一化

        # 对句子特征应用门控
        txt_sent_denoised = txt_sent_denoised * sent_gate_weights  # [b, 512]

        return txt_sent_denoised, txt_word_denoised


# 测试代码
if __name__ == "__main__":
    # 假设batch_size=8，单词维度512，句子维度512
    b = 8
    word_dim = 512
    sent_dim = 512
    max_word_len = 77

    # 随机生成句子级特征和单词级特征
    txt_sent = torch.randn(b, sent_dim)  # [b, 512]
    txt_word = torch.randn(b, max_word_len, word_dim)  # [b, 77, 512]
    
    model = TextDenoisingModule(word_dim, sent_dim)
    txt_sent_denoised, txt_word_denoised = model(txt_sent, txt_word)
    
    print(txt_sent_denoised.shape)  # [b, 512]
    print(txt_word_denoised.shape)  # [b, 77, 512]

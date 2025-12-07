import torch
import torch.nn as nn
import torch.nn.functional as F

class TextDenoisingModule(nn.Module):
    def __init__(self, word_dim, sent_dim, num_heads=8, dropout=0.1):
        super(TextDenoisingModule, self).__init__()
        
        # 单词级别的多头自注意力层
        self.word_attention = nn.MultiheadAttention(embed_dim=word_dim, num_heads=num_heads, dropout=dropout)
        
        # 句子级别的多头自注意力层
        self.sent_attention = nn.MultiheadAttention(embed_dim=sent_dim, num_heads=num_heads, dropout=dropout)
        
        # 任务相关性评分模块，用于提取每个单词和句子的任务相关性
        self.word_importance = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.Tanh(),
            nn.LayerNorm(sent_dim),
            nn.Linear(word_dim, 1)
        )
        self.sent_importance = nn.Sequential(
            nn.Linear(sent_dim, sent_dim),
            nn.Tanh(),
            nn.LayerNorm(sent_dim),
            nn.Linear(sent_dim, 1)
        )

    def forward(self, txt_sent, txt_word):
        # txt_sent: [b, 512]
        # txt_word: [b, 77, 512]
        
        # 转换 txt_word 为 [seq_len, batch_size, embed_dim] 形状
        txt_word = txt_word.permute(1, 0, 2)  # [77, b, 512]
        
        # 1. 单词级别的多头注意力处理
        txt_word_denoised, _ = self.word_attention(txt_word, txt_word, txt_word)  # [77, b, 512]
        
        # 计算每个单词的任务相关性
        word_importance_scores = self.word_importance(txt_word_denoised.permute(1, 0, 2))  # [b, 77, 1]
        word_importance_scores = torch.sigmoid(word_importance_scores.squeeze(-1))  # [b, 77]
        
        # 动态加权：根据任务相关性调整每个单词的重要性
        txt_word_denoised = txt_word_denoised.permute(1, 0, 2)  # [b, 77, 512]
        txt_word_denoised = txt_word_denoised * word_importance_scores.unsqueeze(-1)  # [b, 77, 512]

        # 2. 句子级别的多头注意力处理
        # 转换 txt_sent 为 [seq_len, batch_size, embed_dim] 形状
        txt_sent = txt_sent.unsqueeze(0)  # [1, b, 512]

        txt_sent_denoised, _ = self.sent_attention(txt_sent, txt_sent, txt_sent)  # [1, b, 512]
        txt_sent_denoised = txt_sent_denoised.squeeze(0)  # [b, 512]

        # 计算句子的任务相关性
        sent_importance_scores = self.sent_importance(txt_sent_denoised)  # [b, 1]
        sent_importance_scores = torch.sigmoid(sent_importance_scores)  # [b, 1]

        # 动态加权：根据任务相关性调整整个句子的权重
        txt_sent_denoised = txt_sent_denoised * sent_importance_scores  # [b, 512]

        return txt_sent_denoised, txt_word_denoised

# 测试代码
if __name__ == "__main__":
    # 假设 batch_size = 8，单词维度 = 512，句子维度 = 512
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

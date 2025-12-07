class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim*3, bias = False)
        self.proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, N, C = x.shape
        Q, K, V = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        atten = (Q@K.transpose(-1,-2)) * self.scale
        atten = atten.softmax(dim=-1) 
        X = (atten @ V).reshape(B, N, C)
        X = self.proj(X)
        X = self.dropout(X)
        
        return X

if __name__ == '__main__':
    data = torch.randn(1, 100, 512)
    model = Attention(512, 8, 0.1)
    out = model(data)
    print(out.shape)
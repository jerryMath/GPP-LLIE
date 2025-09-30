import torch

# Two 4D tensors
A = torch.randn(2, 3, 4, 5)  # shape [batch1, batch2, m, k]
B = torch.randn(2, 3, 5, 6)  # shape [batch1, batch2, k, n]

# Matrix multiplication along the last two dims
C = torch.matmul(A, B)

print("A shape:", A.shape)
print("B shape:", B.shape)
print("C shape:", C.shape)


C = A @ B

print("A shape:", A.shape)
print("B shape:", B.shape)
print("C shape:", C.shape)


C = B @ A

print("A shape:", A.shape)
print("B shape:", B.shape)
print("C shape:", C.shape)
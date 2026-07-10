import tensors as ts

# ---------- 1D Tensor ----------
print("=== 1D Tensor ===")
t1 = ts.Tensor([1, 2, 3, 4, 5])
print(t1)                      # Tensor([1, 2, 3, 4, 5], shape=(5,), dtype='float64')
print(t1[0])                   # 1.0
print(t1[1:4])                 # Tensor([2, 3, 4], shape=(3,), dtype='float64')
print(ts.mean(t1))             # 3.0

# ---------- 2D Tensor ----------
print("\n=== 2D Tensor ===")
t2 = ts.Tensor([[1, 2, 3], [4, 5, 6]])
print(t2)

print(t2[0, 0])                # 1.0
print(t2[1, 2])                # 6.0
print(t2[0, :])                # Row 0
print(t2[:, 0])                # Column 0

# ---------- Different dtypes ----------
print("\n=== Different dtypes ===")
t_f32 = ts.Tensor([1, 2, 3], dtype=ts.float32)
print(t_f32)                   # float32
print(f"  dtype name: {t_f32.dtype.name}, bytes: {t_f32.dtype.size}")

t_int = ts.Tensor([1, 2, 3], dtype=ts.int32)
print(t_int)                   # int32
print(f"  dtype name: {t_int.dtype.name}, typecode: {t_int.dtype.typecode}")

# ---------- Operations ----------
print("\n=== Operations ===")
a = ts.Tensor([[1, 2], [3, 4]])
b = ts.Tensor([[5, 6], [7, 8]])

print(a + b)                   # [[6, 8], [10, 12]]
print(a * 2)                   # [[2, 4], [6, 8]]
print(a + 10)                  # [[11, 12], [13, 14]]

# ---------- Matrix Multiplication ----------
print("\n=== Matrix Multiplication ===")
c = ts.Tensor([[1, 2], [3, 4]])
d = ts.Tensor([[5, 6], [7, 8]])
result = ts.dot(c, d)
print(result)

# ---------- Reshape ----------
print("\n=== Reshape ===")
t = ts.Tensor([1, 2, 3, 4, 5, 6])
reshaped = ts.reshape(t, 2, 3)
print(reshaped)

# ---------- Transpose ----------
print("\n=== Transpose ===")
mat = ts.Tensor([[1, 2, 3], [4, 5, 6]])
print(ts.transpose(mat))

# ---------- Statistics ----------
print("\n=== Statistics ===")
data = ts.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Sum: {ts.sum(data)}")     # 55.0
print(f"Mean: {ts.mean(data)}")   # 5.5
print(f"Min: {ts.min(data)}")     # 1.0
print(f"Max: {ts.max(data)}")     # 10.0
print(f"Std: {ts.std(data):.4f}") # 2.8723
1. 张量的秩rank（或者阶），轴（维度）的个数
```python
import numpy as np
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
m.ndim  # 可以直接用ndim求秩
len(m.shape)    # shape返回一个维度的元组，元组的长度就是秩
m.size  # 张量中元素的总个数，等于shape元组中所有元素的乘积
```
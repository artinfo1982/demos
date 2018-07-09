# AI的一些知识点、技巧
## 2D卷积特征图边长的计算公式
```text
输入的正方形边长（side length） = s
卷积核边长（kernel size） = k
步长（stride） = t
补齐（pad） = p
输出的正方形边长 = round((s + 2*p - k) / t) + 1
其中 round 表示向下取整。
```

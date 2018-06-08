# C++的若干技巧

## 模板
例如max函数，通常情况下可以这样定义：
```C
int max(int x, int y) {
	return (x>y)?x:y;
}
float max(float x, float y) {
	return (x>y)?x:y;
}
double max(double x, double y) {
	return (x>y)?x:y;
}
```
但如果我们想使用char a,b; max(a, b);就会报错，因为我们没有定义char max(char x, char y)。能否只写一份代码就解决问题，可以使用模板。   
函数模板的一般形式如下：
```C
template <class或者也可以用typename T>

返回类型 函数名（形参表）
{//函数定义体 }
```

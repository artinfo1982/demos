/*
* 演示各种常用容器的用法
* string、vector、array、forward_list、list
*/

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <forward_list>
#include <list>
#include <functional>

class A
{
  public:
    A(std::string name) : _n(name){};
    std::string &get_name() { return _n; }

  private:
    std::string _n;
};

int main()
{
    /*
    * string，最常用，重点掌握append、replace、find方法
    */
    std::string str1("abcdefg");
    const char *cp = "0123456789";
    std::string str2(cp + 2, 3); // 从cp[2]处拷贝3个字符赋值，"234"
    std::cout << "str1.size: " << str1.size() << std::endl;
    std::cout << "str1.length: " << str1.length() << std::endl;
    std::cout << "str1.max_size: " << str1.max_size() << std::endl;
    std::cout << "str1.at(2): " << str1.at(2) << std::endl;     // 取str1[2]的字符
    std::cout << "str1.front(): " << str1.front() << std::endl; // 取str1的第一个字符
    std::cout << "str1.back(): " << str1.back() << std::endl;   // 取syt1的最后一个字符
    std::cout << "str1[2]: " << str1[2] << std::endl;           // 取str1[2]的字符
    str2.append("haha");                                        // 追加一个字符串
    str2.push_back('\n');                                       // 追加一个字符
    str2.pop_back();                                            // 将最后一个字符删掉
    std::string str3("0123456789");
    std::cout << "str3-1: " << str3 << std::endl;
    str3.erase(2, 3); // 删除"0123456789"从2到4的字符，输出"0156789"
    std::cout << "str3-2: " << str3 << std::endl;
    std::string str4("0000000000");
    std::cout << "str4-1: " << str4 << std::endl;
    str4.replace(2, 5, "111"); // 把"0000000000"从[2]到[5]的所有内容用substr替换，输出"00111000"
    std::cout << "str4-2: " << str4 << std::endl;
    // 注意，find和find_first_of、find_last_of的最大区别在于，find是精确匹配，必须完全包含，而find_first_of、find_last_of只要求部分匹配
    if (str1.find("joke") == std::string::npos) // 正向查找子串，如果有，则返回位置，没有，返回std::string::npos
        std::cout << "str1, find, joke, none" << std::endl;
    else
        std::cout << "str1, find, joke, ok" << std::endl;
    if (str1.rfind("joke") == std::string::npos) // 反向查找子串，如果有，则返回位置，没有，返回std::string::npos
        std::cout << "str1, rfind, joke, none" << std::endl;
    else
        std::cout << "str1, rfind, joke, ok" << std::endl;
    if (str1.find_first_of("cdxyz") == std::string::npos) // 正向查找子串，如果有，则返回位置，没有，返回std::string::npos
        std::cout << "str1, find_first_of, cdxyz, none" << std::endl;
    else
        std::cout << "str1, find_first_of, cdxyz, ok" << std::endl;
    if (str1.find_last_of("cdxyz") == std::string::npos) // 反向查找子串，如果有，则返回位置，没有，返回std::string::npos
        std::cout << "str1, find_last_of, cdxyz, none" << std::endl;
    else
        std::cout << "str1, find_last_of, cdxyz, ok" << std::endl;
    std::cout << "substr of str1: " << str1.substr(2, 3) << std::endl; // substr(pos, len), 如果len不写，则到最后
    std::cout << "-------------------------------------------------------" << std::endl;

    /*
    * vector，优点：动态伸缩，尾部增删元素高效；缺点：在中间增删元素效率低下
    */
    std::vector<std::string> vec1;
    vec1.push_back("123"); // push_back，追加一个元素
    vec1.push_back("abc");
    for (auto &v : vec1)
        std::cout << v << " ";
    std::cout << std::endl;
    std::vector<A> vec2;
    vec2.emplace_back("aaa"); // emplace_back，追加类对象，可以直接调用构造函数对象插入
    vec2.emplace_back("bbb");
    for (auto &v : vec2)
        std::cout << v.get_name() << " ";
    std::cout << std::endl;
    std::vector<std::string> vec3;
    vec3.push_back("111");
    vec3.push_back("222");
    vec3.pop_back(); // pop_back，删除最后一个元素
    for (auto &v : vec3)
        std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "vec1.size: " << vec1.size() << std::endl; // size是当前容器真实占用的大小，也就是容器当前拥有多少个对象
    std::cout << "vec1.max_size: " << vec1.max_size() << std::endl;
    std::vector<int> vec4;
    std::cout << "vec4 capacity: " << vec4.capacity() << std::endl; // capacity是指发生realloc前能允许的最大元素数，即预分配的内存
    vec4.reserve(10);                                            // 通过reserve调整capacity
    std::cout << "vec4 capacity: " << vec4.capacity() << std::endl;
    std::cout << "vec1[1]: " << vec1[1] << std::endl;
    std::cout << "vec1.at(0): " << vec1.at(0) << std::endl;
    std::cout << "vec1.front: " << vec1.front() << std::endl; // 返回首元素
    std::cout << "vec1.back: " << vec1.back() << std::endl;   // 返回末元素
    if (vec1.empty())
        std::cout << "vec1 is empty" << std::endl;
    else
        std::cout << "vec1 is not empty" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    /*
    * array，大小固定，c++11引入，优点：快速随机访问；缺点：不能弹性伸缩
    * array比内置数组的优势在于提供了front、back、at的安全访问元素的方法，而数组只能通过下标，容易产生越界
    * 注意，array对于一维数组比较合适，如果定义高维数组，建议采用内置数组更加高效，形如float a[2][3][4]
    */
    std::array<int, 3> arr1;
    arr1.fill(1); // 给array的所有元素都赋值为1
    std::array<std::string, 3> arr2 = {"111", "222", "333"};
    if (arr2.empty())
        std::cout << "arr2 is empty" << std::endl;
    else
        std::cout << "arr2 is not empty" << std::endl;
    std::cout << "arr2 size: " << arr2.size() << std::endl;
    std::cout << "arr2 max_size: " << arr2.max_size() << std::endl;
    std::cout << "arr2[1]: " << arr2[1] << std::endl;
    std::cout << "arr2.at(0): " << arr2.at(0) << std::endl;
    std::cout << "arr2.front: " << arr2.front() << std::endl; // 返回首元素
    std::cout << "arr2.back: " << arr2.back() << std::endl;   // 返回末元素
    for (auto &a : arr2)
        std::cout << a << " ";
    std::cout << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    /*
    * forward_list，单向链表，优势：快速在中间增删元素；缺点：随机访问效率低
    * 注意：insert_after、emplace_after是在指定位置的后面插入元素
    */
    std::forward_list<std::string> fl1;
    fl1.push_front("333"); // 最前端插入一个元素
    fl1.push_front("111");
    fl1.insert_after(fl1.begin(), "222"); // 在某一位的后面插入一个元素
    for (auto &f : fl1)
        std::cout << f << " ";
    std::cout << std::endl;
    std::forward_list<A> fl2;
    fl2.emplace_front("ccc");
    fl2.emplace_front("aaa");
    fl2.emplace_after(fl2.begin(), "bbb");
    for (auto &f : fl2)
        std::cout << f.get_name() << " ";
    std::cout << std::endl;
    std::cout << "fl1 max_size: " << fl1.max_size() << std::endl;
    if (fl1.empty())
        std::cout << "fl1 is empty" << std::endl;
    fl2.pop_front();              // 删除首元素
    fl2.erase_after(fl2.begin()); // 删除某位置后的一个元素
    for (auto &f : fl2)
        std::cout << f.get_name() << " ";
    std::cout << std::endl;
    std::forward_list<int> fl3 = {1, 100, 2, 3, 10, 1, 11, -1, 12, 12};
    fl3.remove(1);                               // 删除两个等于1的元素
    fl3.remove_if([](int n) { return n > 10; }); // 移除全部大于10的元素
    fl3.unique();                                // 剔除所有重复的元素
    for (auto &f : fl3)
        std::cout << f << " ";
    std::cout << std::endl;
    std::forward_list<int> fl4 = {3, 6, 1, 4, 9};
    std::forward_list<int> fl5 = {2, 9, 1, 7, 4};
    fl4.sort();                    // 从小到大排序
    fl5.sort(std::greater<int>()); // 从大到小排序，std::greater在<functional>中定义
    fl4.merge(fl5);                // 将fl5并入fl4
    for (auto &f : fl4)
        std::cout << f << " ";
    std::cout << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    /*
    * list，双向链表，优势：快速在中间增删元素；缺点：随机访问效率低
    * 注意：insert、emplace是在指定位置的前面插入元素
    */
    std::list<std::string> li1;
    li1.push_front("333"); // 最前端插入一个元素
    li1.push_front("111");
    li1.insert(++begin(li1), "222"); // 在某一位的前面插入一个元素
    li1.push_back("444");             // 在末尾插入一个元素
    for (auto &l : li1)
        std::cout << l << " ";
    std::cout << std::endl;
    std::list<A> li2;
    li2.emplace_front("ccc");
    li2.emplace_front("aaa");
    li2.emplace(++begin(li2), "bbb"); // 在某一位的前面插入一个元素
    li2.emplace_back("ddd");          // 在末尾插入一个类对象
    for (auto &l : li2)
        std::cout << l.get_name() << " ";
    std::cout << std::endl;
    std::cout << "li1 size: " << li1.size() << std::endl;
    std::cout << "li1 max_size: " << li1.max_size() << std::endl;
    if (li1.empty())
        std::cout << "li1 is empty" << std::endl;
    std::cout << "li1.front: " << li1.front() << std::endl; // 返回首元素
    std::cout << "li1.back: " << li1.back() << std::endl;   // 返回末元素
    li2.pop_front();                                        // 删除首元素
    li2.pop_back();                                         // 删除末尾元素
    li2.erase(++begin(li2));
    for (auto &l : li2)
        std::cout << l.get_name() << " ";
    std::cout << std::endl;
    // remove、remove_if、unique、sort、merge同forward_list，不再赘述
    std::cout << "-------------------------------------------------------" << std::endl;

    return 0;
}
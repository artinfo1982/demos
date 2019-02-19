/*
* 智能指针实质是一个对象，行为表现的却像一个指针。
* shared_ptr和unique_ptr之间的区别在于：shared_ptr是引用计数的智能指针，而unique_ptr不是。
* 这意味着，可以有多个shared_ptr实例指向同一块动态分配的内存，当最后一个shared_ptr离开作用域时，才会释放这块内存。
* shared_ptr也是线程安全的。unique_ptr离开作用域时，会立即释放底层内存。
* 默认的智能指针应该是unique_ptr。只有需要共享资源时，才使用shared_ptr。
* weak_ptr并不拥有其指向的对象，也就是说，让weak_ptr指向shared_ptr所指向的对象，对象的引用计数并不会增加。
* weak_ptr并不持有对象，也就无法访问对象，但是可以通过weak_ptr获取对应的shared_ptr，从而访问对象，weak_ptr充当桥梁的作用。
*/

#include <iostream>
#include <vector>
#include <map>
#include <memory>

class A
{
  public:
    A()
    {
        mem = new char[10];
    }
    ~A()
    {
        delete[] mem;
    }

  private:
    char *mem;
};

int main()
{
    auto shared_array = std::make_shared<std::array<int, 10>>();
    for (int i = 0; i < 10; ++i)
        shared_array.get()->at(i) = i;
    for (int i = 0; i < 10; ++i)
        std::cout << shared_array.get()->at(i) << " ";
    std::cout << std::endl;

    auto unique_array = std::unique_ptr<int>(new int[10]); // unique_ptr，有默认的default_delete
    for (int i = 0; i < 10; ++i)
        unique_array.get()[i] = i;
    for (int i = 0; i < 10; ++i)
        std::cout << unique_array.get()[i] << " ";
    std::cout << std::endl;

    auto vec1 = std::make_shared<std::vector<int>>();
    vec1->push_back(1);
    vec1->push_back(2);
    for (auto a : *vec1)
        std::cout << a << std::endl;

    std::unique_ptr<std::vector<int>> vec2(new std::vector<int>);
    vec2->push_back(1);
    vec2->push_back(2);
    for (auto a : *vec2)
        std::cout << a << std::endl;

    std::shared_ptr<std::map<int, std::string>> map1(new std::map<int, std::string>); // 虽然shared_ptr可以用new，但不建议这么使用，尽量用make_shared
    map1->insert(std::make_pair(1, "aaa"));
    map1->insert(std::make_pair(2, "bbb"));
    for (auto a : *map1)
        std::cout << "key=" << a.first << ", value=" << a.second << std::endl;

    std::unique_ptr<std::map<int, std::string>> map2(new std::map<int, std::string>); // 而unique_ptr，因为在c++14之后才出现make_unique，在c++11只能用new
    map2->insert(std::make_pair(1, "aaa"));
    map2->insert(std::make_pair(2, "bbb"));
    for (auto a : *map2)
        std::cout << "key=" << a.first << ", value=" << a.second << std::endl;

    auto sp1 = std::make_shared<A>();              // 创建单个类对象shared_ptr
    std::unique_ptr<A> up1(new A());               // 创建单个类对象unique_ptr
    auto sp2 = std::make_shared<std::vector<A>>(); // 创建类对象数组shared_ptr
    std::unique_ptr<A[]> up2(new A[10]); // 创建类对象数组unique_ptr

    auto sp3 = std::make_shared<int>(10);
    std::weak_ptr<int> wp1(sp3);
    if (!wp1.expired()) // 判断是否被析构
        if (std::shared_ptr<int> sp4 = wp1.lock()) // get shared_ptr
            std::cout << "good" << std::endl;
    
    return 0;
}
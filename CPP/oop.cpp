/*
* CPP 面向对象的一些示例
*/

#include <iostream>
#include <memory>

/* 阻止继承 */
class Z final // final表示此类不能被继承
{
  public:
    Z() = default;
};

/* 如下代码错误，因为Z已经是final类型，禁止被继承
class Z1 : public Z
{
  public:
    Z1() = default;
};
*/

/* 继承、虚函数、静态成员变量、静态成员方法 */
class A
{
  public:
    A() = default;
    A(int n) : _n(n)
    {
        std::cout << "A constructor" << std::endl;
    }
    virtual ~A();            // 析构函数也是虚函数，需要派生类自己去override
    virtual int get() const; // 虚函数，需要派生类自己去override
    static int s_get();

  private:
    int _n;
    static int _sn;
};

class B : public A
{
  public:
    B() = default;
    B(int n) : _n(n) { std::cout << "B constructor" << std::endl; }
    ~B() override; // override关键字表示派生类的虚函数，如果标记了override，而基类却没有这样的虚函数声明，则报错
    int get() const override;

  private:
    int _n;
};

/*
* 纯虚函数、抽象类、接口
* 只要含有至少一个纯虚函数的类，就是抽象类，抽象类不能实例化，只能被继承
* 抽象类最大的作用，就是对接口(体现为纯虚函数)的封装
*/
class C
{
  public:
    C() = default;
    virtual void f1() = 0; // 纯虚函数（接口），所有派生自抽象类的类，都必须override此方法
};

class D : public C
{
  public:
    D() = default;
    void f1() override;
};

/* 类的访问控制（public、protected、private） */
class E
{
  public:
    E() = default; // 在有继承的情况下，基类必须提供默认的构造函数
    E(int n1, int n2, int n3) : e_public(n1), e_protected(n2), e_private(n3) {}
    int e_public;
    int e_public_get_public() { return e_public; }
    int e_public_get_protected() { return e_protected; }
    int e_public_get_private() { return e_private; }

  protected:
    int e_protected;
    int e_protected_get_protected() { return e_protected; }

  private:
    int e_private;
    int e_private_get_private() { return e_private; }
};

class F : public E
{
  public:
    F(int n1, int n2, int n3) : f_public(n1), f_protected(n2), f_private(n3) {}
    // 子类可以通过函数的方式，访问父类的protected成员，但不能访问private成员
    int f_public;
    int f_get_father_protected() { return e_protected; }

  protected:
    int f_protected;

  private:
    int f_private;
};

class G : protected E
{
  public:
    G(int n1, int n2, int n3) : g_public(n1), g_protected(n2), g_private(n3) {}
    int g_public;
    int g_get_father_protected() { return e_protected; }

  protected:
    int g_protected;

  private:
    int g_private;
};

class H : private E
{
  public:
    H(int n1, int n2, int n3) : h_public(n1), h_protected(n2), h_private(n3) {}
    int h_public;
    int h_get_father_protected() { return e_protected; }

  protected:
    int h_protected;

  private:
    int h_private;
};

int A::_sn = 1; // 类内private的static成员变量初始化，必须在类外进行，也不能在main中操作

A::~A()
{
    std::cout << "A destructor" << std::endl;
}

int A::get() const
{
    std::cout << "A get()" << std::endl;
    return _n;
}

int A::s_get()
{
    std::cout << "A static s_get()" << std::endl;
    return _sn;
}

B::~B()
{
    std::cout << "B destructor" << std::endl;
}

int B::get() const
{
    std::cout << "B get()" << std::endl;
    return _n;
}

void D::f1()
{
    std::cout << "D f1()" << std::endl;
}

int main()
{
    A a1(1);
    a1.get();
    a1.s_get();
    B b1(1);
    b1.get();
    b1.s_get(); // 重要，派生类，可以调用基类的static方法（public、protected）
    A a2 = B(1);
    a2.get();
    /*
    B b2 = A(1); // 错误，不能将基类向派生类转换
    */
    A *a3 = new B(1); // new返回的是一个指向派生类对象的指针
    a3->get();        // 既然是指向派生类对象的指针，调用的也是派生类的方法
    A a4 = (A)b1;     // 可以显式采用类型强转，但不建议这么用，建议用智能指针
    a4.get();
    std::unique_ptr<A> a5(new B(1));
    a5->get();

    D *d = new D();
    d->f1();

    E *e = new E(1, 2, 3);
    e->e_public;
    /*
    e->e_protected; // 错误，即使是本类对象，也不能访问protected变量
    e->e_private; // 错误，即使是本类对象，也不能访问private变量
    */
    e->e_public_get_public();
    e->e_public_get_protected();
    e->e_public_get_private();
    /*
    e->e_protected_get_protected(); // 错误，即使是本类对象，也不能访问protected方法
    e->e_private_get_private(); // 错误，即使是本类对象，也不能访问private方法
    */
    F *f = new F(4, 5, 6);
    f->e_public;
    /*
    f->e_protected; // 错误，public继承的子类对象不能访问父类的protected对象
    f->e_private; // 错误，public继承的子类对象不能访问父类的private对象
    */
    /* 错误，父类对象，无法访问public继承的子类的所有成员变量
    e->f_public;
    e->f_protected;
    e->f_private;
    */

    G *g = new G(7, 8, 9);
    /* 错误，protected继承的子类对象，无法访问父类的所有成员变量
    g->e_public;
    g->e_protected;
    g->e_private
    */
    /* 错误，父类对象，无法访问protected继承的子类的所有成员变量
    e->g_public;
    e->g_protected;
    e->g_private;
    */

    H *h = new H(10, 11, 12);
    /* 错误，private继承的子类对象，无法访问父类的所有成员变量
    h->e_public;
    h->e_protected;
    h->e_private;
    */
    /* 错误，父类对象，无法访问private继承的子类的所有成员变量
    e->h_public;
    e->h_protected;
    e->h_private;
    */

    f->f_get_father_protected();
    g->g_get_father_protected();
    h->h_get_father_protected();

    return 0;
}
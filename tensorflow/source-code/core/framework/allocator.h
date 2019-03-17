#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_

#include <stdlib.h>

#include <limits>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow
{

class Variant;

// 分配内存行为的属性
struct AllocationAttributes
{
    // 分配失败是否需要重试
    bool no_retry_on_failure = false;
    // 是否记录分配行为日志
    bool allocation_will_be_logged = false;
    // 试验性属性，略
    std::function<uint64()> freed_by_func = nullptr;
};

// 运行时分配器统计收集的信息
struct AllocatorStats
{
    int64 num_allocs;         // 分配了多少次
    int64 bytes_in_use;       // 使用中的bytes
    int64 peak_bytes_in_use;  // 使用中的峰值bytes
    int64 largest_alloc_size; // 最大单次分配的bytes
    // 用户能分配的最大字节数（如果可以获知的话）
    absl::optional<int64> bytes_limit;
    // 结构体初始化，类似class初始化
    AllocatorStats()
        : num_allocs(0),
          bytes_in_use(0),
          peak_bytes_in_use(0),
          largest_alloc_size(0) {}
    string DebugString() const;
};

// 定义分配和释放内存的抽象接口类
class Allocator
{
  public:
    static constexpr size_t kAllocatorAlignment = 64; // 64字节对齐
    virtual ~Allocator();
    virtual string Name() = 0; // 分配器的名字
    // 返回num_bytes字节的未初始化内存块，raw表示未初始化，按照alignment对齐，alignment必须是2的幂次
    virtual void *AllocateRaw(size_t alignment, size_t num_bytes) = 0;
    // 返回num_bytes字节的未初始化内存块，raw表示未初始化，按照alignment对齐，alignment必须是2的幂次
    virtual void *AllocateRaw(size_t alignment, size_t num_bytes, const AllocationAttributes &allocation_attr)
    {
        return AllocateRaw(alignment, num_bytes); // 默认行为
    }
    // 释放ptr指向的内存块，ptr!=nullptr
    virtual void DeallocateRaw(void *ptr) = 0;
    // 泛化的分配接口，AllocationAttributes是一个结构体，AllocationAttributes()是初始化一个结构体并返回其对象引用
    // struct A{...}; A()是初始化这个结构体，并返回隐式对象引用
    template <typename T>
    T *Allocate(size_t num_elements)
    {
        return Allocate<T>(num_elements, AllocationAttributes());
    }
    // 分配内存主入口
    template <typename T>
    T *Allocate(size_t num_elements, const AllocationAttributes &allocation_attr)
    {
        // 如果需要分配的内存太大，以至于超过了size_t的最大值，直接返回NULL
        if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T)))
            return NULL;
        // 分配一块未初始化的内存块
        void *p = AllocateRaw(kAllocatorAlignment, sizeof(T) * num_elements, allocation_attr);
        // 将void*转换为特定的类型
        T *typed_p = reinterpret_cast<T *>(p);
        // 如果分配成功，则调用相应的构造函数来填充已分配的内存块
        if (typed_p)
            RunCtor<T>(typed_p, num_elements);
        return typed_p;
    }
    // 泛化的释放内存接口
    template <typename T>
    void Deallocate(T *ptr, size_t num_elements)
    {
        if (ptr)
        {
            RunDtor<T>(ptr, num_elements);
            DeallocateRaw(ptr);
        }
    }
    // 是否能跟踪到分配内存的大小
    virtual bool TracksAllocationSizes() { return false; }
    // 是否需要分配空张量（特殊测试需要）
    virtual bool ShouldAllocateEmptyTensors() { return false; }
    // 返回ptr指向的用户需要分配的内存大小，ptr!=nullptr
    virtual size_t RequestedSize(const void *ptr)
    {
        CHECK(false) << "allocator doesn't track sizes";
        return size_t(0);
    }
    // 返回ptr指向的实际已分配内存的大小，ptr!=nullptr
    virtual size_t AllocatedSize(const void *ptr) { return RequestedSize(ptr); }
    // 返回64位int类型的已分配内存块ID
    virtual int64 AllocationId(const void *ptr) { return 0; }
    // 同AllocatedSize，但可能执行很慢
    virtual size_t AllocatedSizeSlow(const void *ptr)
    {
        if (TracksAllocationSizes())
            return AllocatedSize(ptr);
        return 0;
    }
    // 获取内存分配器的状态
    virtual absl::optional<AllocatorStats> GetStats() { return absl::nullopt; }
    // 清除内存分配器的状态
    virtual void ClearStats() {}

  private:
    // 以下这些原子构造函数和析构函数，不能被直接调用
    template <typename T>
    void RunCtor(T *p, size_t n)
    {
        // static_assert(bool, string)，是静态断言，可以在编译期就进行断言，判断T是不是基本类型，比如int之类的
        static_assert(is_simple_type<T>::value, "T is not a simple type.");
    }
    template <typename T>
    void RunDtor(T *p, size_t n) {}
    // 逐个运行string数组中的每个string的构造函数，p[0], p[1], ..., p[n-1].
    virtual void RunStringCtor(string *p, size_t n)
    {
        for (size_t i = 0; i < n; ++p, ++i)
            new (p) string();
    }
    // 逐个运行string数组中的每个string的构造函数，p[0], p[1], ..., p[n-1].
    virtual void RunStringDtor(string *p, size_t n)
    {
        for (size_t i = 0; i < n; ++p, ++i)
            p->~string();
    }
    virtual void RunResourceCtor(ResourceHandle *p, size_t n)
    {
        for (size_t i = 0; i < n; ++p, ++i)
            new (p) ResourceHandle();
    }
    virtual void RunResourceDtor(ResourceHandle *p, size_t n)
    {
        for (size_t i = 0; i < n; ++p, ++i)
            p->~ResourceHandle();
    }
    virtual void RunVariantCtor(Variant *p, size_t n);
    virtual void RunVariantDtor(Variant *p, size_t n);
};

// Allocator-specific constructors and destructors are used for
// strings
template <>
inline void Allocator::RunCtor(string *p, size_t n)
{
    RunStringCtor(p, n);
}

template <>
inline void Allocator::RunDtor(string *p, size_t n)
{
    RunStringDtor(p, n);
}

template <>
inline void Allocator::RunCtor(ResourceHandle *p, size_t n)
{
    RunResourceCtor(p, n);
}

template <>
inline void Allocator::RunDtor(ResourceHandle *p, size_t n)
{
    RunResourceDtor(p, n);
}

template <>
inline void Allocator::RunCtor(Variant *p, size_t n)
{
    RunVariantCtor(p, n);
}

template <>
inline void Allocator::RunDtor(Variant *p, size_t n)
{
    RunVariantDtor(p, n);
}

// 封装了Allocator
class AllocatorWrapper : public Allocator
{
  public:
    /*
    explicit的作用是防止类构造函数发生隐式类型转换
    例如：
    class A {
        public:
            A(int i): _i(i) {}
        private:
            int _i;
    };
    A a = 1; // 相当于tmp = A(1); A a(tmp); tmp.~A();
    一旦变成下面的样子：
        public:
            explicit A(int i): _i(i) {}
    则 A a = 1; // 报错
    只能显式调用： A a = A(1);
    */
    explicit AllocatorWrapper(Allocator *wrapped) : wrapped_(wrapped) {}

    ~AllocatorWrapper() override {}

    // Returns the wrapped allocator to which all calls are delegated.
    Allocator *wrapped() const { return wrapped_; }
    string Name() override { return wrapped_->Name(); }
    void *AllocateRaw(size_t alignment, size_t num_bytes) override
    {
        return wrapped_->AllocateRaw(alignment, num_bytes);
    }
    void *AllocateRaw(size_t alignment, size_t num_bytes, const AllocationAttributes &allocation_attr) override
    {
        return wrapped_->AllocateRaw(alignment, num_bytes, allocation_attr);
    }
    void DeallocateRaw(void *ptr) override { wrapped_->DeallocateRaw(ptr); }
    bool TracksAllocationSizes() override
    {
        return wrapped_->TracksAllocationSizes();
    }
    bool ShouldAllocateEmptyTensors() override
    {
        return wrapped_->TracksAllocationSizes();
    }
    size_t RequestedSize(const void *ptr) override
    {
        return wrapped_->RequestedSize(ptr);
    }
    size_t AllocatedSize(const void *ptr) override
    {
        return wrapped_->AllocatedSize(ptr);
    }
    int64 AllocationId(const void *ptr) override
    {
        return wrapped_->AllocationId(ptr);
    }
    size_t AllocatedSizeSlow(const void *ptr) override
    {
        return wrapped_->AllocatedSizeSlow(ptr);
    }

  private:
    Allocator *const wrapped_;
};

/*
分配器自身的属性。
出于性能考虑，有时需要向不同的设备申请内存，比如在使用GPU时也需要向CPU申请内存，
这时就需要在分配内存时，指定内存的属性。
示例：
    普通设备内存分配器：
    Allocator* a = allocator(AllocatorAttributes());
    分配CPU RAM，不管算子在哪里执行：
    AllocatorAttributes attr;
    attr.set_on_host(true);
    Allocator* a = allocator(attr);
*/
struct AllocatorAttributes
{
    void set_on_host(bool v) { value |= (static_cast<int>(v)); } // v-->int
    bool on_host() const { return value & 0x1; }                 // value-->bool
    // 左移1位，x2
    void set_nic_compatible(bool v) { value |= (static_cast<int>(v) << 1); }
    bool nic_compatible() const { return value & (0x1 << 1); } // value & 0x10
    // 左移2位，x4
    void set_gpu_compatible(bool v) { value |= (static_cast<int>(v) << 2); }
    bool gpu_compatible() const { return value & (0x1 << 2); } // value & 0x100
    void Merge(AllocatorAttributes other)
    {
        value |= other.value;
        scope_id = (scope_id > 0 && other.scope_id == 0)
                       ? scope_id
                       : ((scope_id == 0) ? other.scope_id : 0);
    }
    // 比较不同AllocatorAttributes里的value的大小
    bool IsEqualOrLessRestrictiveThan(const AllocatorAttributes &other) const
    {
        return (value | other.value) == other.value;
    }
    // value，4字节，其中高8bit是和具体的设备对应的
    uint32 value = 0;
    // （试验性），如果scope_id > 0，则分配降级为在相同设备上的特殊用途的分配器
    int32 scope_id = 0;
};

// Returns a trivial implementation of Allocator, which is a process singleton.
// Access through this function is only intended for use by restricted parts
// of the infrastructure.
Allocator *cpu_allocator_base();

// If available, calls ProcessState::GetCPUAllocator(numa_node).
// If not, falls back to cpu_allocator_base().
// Intended for use in contexts where ProcessState is not visible at
// compile time. Where ProcessState is visible, it's preferable to
// call it directly.
Allocator *cpu_allocator(int numa_node = port::kNUMANoAffinity);

// If 'enable' is true, the default CPU allocator implementation will collect
// AllocatorStats. By default, it's disabled.
void EnableCPUAllocatorStats(bool enable);
bool CPUAllocatorStatsEnabled();

// If 'enable' is true, the default CPU allocator implementation will collect
// full statistics. By default, it's disabled.
void EnableCPUAllocatorFullStats(bool enable);
bool CPUAllocatorFullStatsEnabled();

// An object that does the underlying suballoc/free of memory for a higher-level
// allocator.  The expectation is that the higher-level allocator is doing some
// kind of cache or pool management so that it will call SubAllocator::Alloc and
// Free relatively infrequently, compared to the number of times its own
// AllocateRaw and Free methods are called.
class SubAllocator
{
  public:
    // Visitor gets called with a pointer to a memory area and its
    // size in bytes.  The index value will be numa_node for a CPU
    // allocator and GPU id for a GPU allocator.
    typedef std::function<void(void *, int index, size_t)> Visitor;

    SubAllocator(const std::vector<Visitor> &alloc_visitors,
                 const std::vector<Visitor> &free_visitors);

    virtual ~SubAllocator() {}
    virtual void *Alloc(size_t alignment, size_t num_bytes) = 0;
    virtual void Free(void *ptr, size_t num_bytes) = 0;

  protected:
    // Implementation of Alloc() method must call this on newly allocated
    // value.
    void VisitAlloc(void *ptr, int index, size_t num_bytes);

    // Implementation of Free() method must call this on value to be
    // freed immediately before deallocation.
    void VisitFree(void *ptr, int index, size_t num_bytes);

    const std::vector<Visitor> alloc_visitors_;
    const std::vector<Visitor> free_visitors_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_H_
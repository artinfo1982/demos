import com.alibaba.fastjson.JSON;
import java.util.ArrayList;
import java.util.List;

class A {
    public String a;
    public List<B> list = new ArrayList<>();
    public void setA (String a) {
        this.a = a;
    }
}

class B {
    public int b;
    public void setB(int b) {
        this.b = b;
    }
}

public class Test {

    public static void main(String[] args) {

        //序列化
        B b1 = new B();
        B b2 = new B();
        b1.setB(2);
        b2.setB(3);
        A a1 = new A();
        a1.setA("1");
        a1.list.add(b1);
        a1.list.add(b2);
        System.out.println(JSON.toJSONString(a1));

        //反序列化
        String s = "{\"a\":\"4\",\"list\":[{\"b\":5},{\"b\":6}]}";
        A a2 = JSON.parseObject(s, A.class);
        System.out.println("a=" + a2.a);
        for (B b : a2.list) {
            System.out.println("b=" + b.b);
        }
    }
}

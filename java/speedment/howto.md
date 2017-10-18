# speedment项目地址 #
https://github.com/speedment/speedment/   
speedment是一个实现了java ORM的框架，可以使用GUI图形界面连接到数据库，自动生成代码。

# speedment使用方法 #
在idea中创建maven工程，生成pom.xml文件后，在其中追加如下内容：   
```xml
<properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>

    <build>
        <plugins>
            <plugin>
                <groupId>com.speedment</groupId>
                <artifactId>speedment-maven-plugin</artifactId>
                <version>3.0.15</version>
            </plugin>
        </plugins>
    </build>

    <dependencies>
        <dependency>
            <groupId>com.speedment</groupId>
            <artifactId>runtime</artifactId>
            <version>3.0.15</version>
            <type>pom</type>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>5.1.44</version>
            <scope>runtime</scope>
        </dependency>
    </dependencies>
```
hello

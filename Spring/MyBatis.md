### **MyBatis 核心机制**

------

#### **1. ORM 对比**

**🔑 知识点详解**

- **ORM（Object-Relational Mapping，对象关系映射）**：

  - 通过对象模型（Java 类）与数据库表之间的映射，简化数据库操作。
  - 主要目标：减少 SQL 编写、降低数据库操作的复杂性。

- **MyBatis vs. Hibernate（常见 ORM 框架对比）**：

  | 特性     | MyBatis            | Hibernate                    |
  | -------- | ------------------ | ---------------------------- |
  | SQL 控制 | 需要手写 SQL       | 自动生成 SQL                 |
  | 灵活性   | 高，可优化 SQL     | 低，依赖 HQL 或 Criteria API |
  | 性能     | 适用于复杂查询     | 适用于 CRUD 和标准查询       |
  | 缓存机制 | 提供一级、二级缓存 | 提供更完善的缓存管理         |
  | 事务控制 | 依赖手动管理       | 内置事务管理                 |

  👉 **注意**：

  - **MyBatis 适合复杂 SQL 逻辑的应用**，如数据量大、SQL 需优化的场景。
  - **Hibernate 适用于标准 CRUD 操作较多的应用**，可减少 SQL 编写工作量。

**🔥 面试高频题**

1. MyBatis 与 Hibernate 的区别？
   - **一句话答案**：MyBatis 需要手写 SQL，适用于复杂查询；Hibernate 自动生成 SQL，更适合标准 CRUD 操作。
   - 深入回答：
     - MyBatis 提供灵活的 SQL 操作，避免 Hibernate 可能存在的 SQL 性能问题。
     - Hibernate 采用全自动 ORM 机制，减少开发工作量，但可能会带来 SQL 优化难度。

------

#### **2. MyBatis 核心流程**

**🔑 知识点详解**

- **MyBatis 工作流程**

  1. **加载配置文件**：解析 `mybatis-config.xml` 和 `mapper.xml`。
  2. **创建 SqlSessionFactory**：构建会话工厂，用于管理数据库会话。
  3. **获取 SqlSession**：开启数据库会话，执行 SQL 语句。
  4. **执行 SQL 语句**：调用 `selectList()`、`selectOne()`、`insert()`、`update()`、`delete()`。
  5. **映射结果**：将查询结果映射到 Java 对象。
  6. **关闭 SqlSession**：释放数据库资源。

  **示例代码**

  ```java
  InputStream inputStream = Resources.getResourceAsStream("mybatis-config.xml");
  SqlSessionFactory sqlSessionFactory = new SqlSessionFactoryBuilder().build(inputStream);
  try (SqlSession session = sqlSessionFactory.openSession()) {
      UserMapper mapper = session.getMapper(UserMapper.class);
      User user = mapper.getUserById(1);
  }
  ```

  👉 **注意**：

  - `SqlSessionFactory` 线程安全，推荐单例模式管理。
  - `SqlSession` 不是线程安全的，**每次数据库操作需获取新会话，避免资源泄露**。

**🔥 面试高频题**

1. MyBatis 的执行流程是怎样的？
   - **一句话答案**：MyBatis 通过 `SqlSessionFactory` 创建 `SqlSession`，加载 SQL 语句并映射结果对象。
   - 深入回答：
     - 解析 XML 配置，建立数据库连接池，执行 SQL，映射结果，最后释放资源。
     - 推荐使用 **Mapper 接口绑定**，减少 `sqlSession` 的直接调用。

------

#### **3. MyBatis 缓存机制**

**🔑 知识点详解**

- MyBatis 提供两级缓存：

  1. 一级缓存（SqlSession 级别）

     - 默认开启，每个 `SqlSession` 维护独立的缓存。

     - 作用范围：仅限于同一个 `SqlSession`，关闭后缓存失效。

     - 示例：

       ```java
       try (SqlSession session = sqlSessionFactory.openSession()) {
           User user1 = session.selectOne("getUserById", 1);
           User user2 = session.selectOne("getUserById", 1); // 直接从缓存获取
       }
       ```

  2. 二级缓存（Mapper 级别）

     - 需要手动开启，多个 `SqlSession` 共享。

     - 作用范围：相同 `namespace` 共享缓存，不同 `namespace` 之间不共享。

     - 开启方式：

       ```xml
       <cache/>
       ```

     - 示例：

       ```java
       sqlSessionFactory.getConfiguration().addMapper(UserMapper.class);
       ```

  👉 注意：

  - **一级缓存默认开启，不同 SqlSession 之间不共享**。
  - **二级缓存需要手动开启，多个 SqlSession 可共享**。
  - **缓存会在 `commit()` 或 `close()` 时生效**。

**🔥 面试高频题**

1. MyBatis 的缓存机制是什么？
   - **一句话答案**：MyBatis 提供一级缓存（SqlSession 级别，默认开启）和二级缓存（Mapper 级别，需手动开启）。
   - 深入回答：
     - 一级缓存局限于同一 `SqlSession`，二级缓存可跨 `SqlSession` 共享。
     - 二级缓存可以自定义实现，如使用 Redis、EhCache 进行持久化存储。

------

#### **4. 动态 SQL 与插件**

**🔑 知识点详解**

- **动态 SQL**

  - 主要作用：根据不同的查询条件，动态生成 SQL 语句，避免硬编码。

  - 常见动态 SQL 语法：

    | 标签                                  | 作用                          |
    | ------------------------------------- | ----------------------------- |
    | `<if>`                                | 条件判断                      |
    | `<choose>` / `<when>` / `<otherwise>` | 选择逻辑                      |
    | `<where>`                             | 自动添加 `WHERE` 关键字       |
    | `<set>`                               | 处理 `UPDATE` 语句的 SET 部分 |
    | `<foreach>`                           | 处理集合遍历                  |

  **示例**

  ```xml
  <select id="getUserByCondition" resultType="User">
      SELECT * FROM users
      <where>
          <if test="name != null">
              name = #{name}
          </if>
          <if test="age != null">
              AND age = #{age}
          </if>
      </where>
  </select>
  ```

- **插件机制**

  - 作用：MyBatis 允许使用插件在 SQL 执行过程中拦截某些方法，如：
    - **拦截 Executor**：控制 SQL 执行流程
    - **拦截 StatementHandler**：修改 SQL 语句
    - **拦截 ResultSetHandler**：处理查询结果

  **示例**

  ```java
  @Intercepts({@Signature(
      type = StatementHandler.class,
      method = "prepare",
      args = {Connection.class, Integer.class}
  )})
  public class MyInterceptor implements Interceptor {
      @Override
      public Object intercept(Invocation invocation) throws Throwable {
          System.out.println("拦截 SQL 语句执行");
          return invocation.proceed();
      }
  }
  ```

  👉 **注意**：

  - **插件通过 `@Intercepts` 注解拦截指定方法**，可用于日志记录、SQL 优化等场景。

**🔥 面试高频题**

1. **MyBatis 的动态 SQL 有哪些关键标签？**
   - **一句话答案**：常见动态 SQL 标签有 `<if>`、`<choose>`、`<where>`、`<set>`、`<foreach>`。
   - 深入回答：
     - `<if>` 用于条件判断，`<foreach>` 适用于 `IN` 查询，`<where>` 可自动添加 `WHERE` 关键字。
2. **MyBatis 的插件机制是如何工作的？**
   - **一句话答案**：MyBatis 允许使用拦截器（Interceptor）在 SQL 执行过程中修改 SQL、优化查询、日志监控等。
   - 深入回答：
     - 插件通过 `@Intercepts` 注解拦截 `Executor`、`StatementHandler` 等组件，可扩展 MyBatis 的功能，如 SQL 日志、性能优化。

------

**🌟 重点总结**

- **MyBatis 适用于复杂 SQL 处理，Hibernate 适用于标准 ORM 场景**
- **MyBatis 执行流程：配置加载 → 创建 SqlSessionFactory → 绑定 Mapper → 执行 SQL**
- **提供一级缓存（默认）和二级缓存（需手动开启）**
- **动态 SQL 通过 `<if>`、`<foreach>` 等标签构造灵活查询**
- **插件机制可拦截 SQL 执行，实现自定义逻辑**
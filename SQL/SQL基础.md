### SQL 基础与高级用法

---

#### **1. 基础查询：SELECT、WHERE、GROUP BY、HAVING、ORDER BY**
**🔑 知识点详解**
- **SELECT**：
  - **定义**：用于从数据库中选择数据。
  - **核心语法**：
    ```sql
    SELECT column1, column2 FROM table_name;
    ```
  - **示例**：
    ```sql
    SELECT name, age FROM users;
    ```

- **WHERE**：
  - **定义**：用于过滤记录，指定条件。
  - **核心语法**：
    ```sql
    SELECT column1, column2 FROM table_name WHERE condition;
    ```
  - **示例**：
    ```sql
    SELECT name, age FROM users WHERE age > 30;
    ```

- **GROUP BY**：
  - **定义**：用于将结果集按一个或多个列进行分组。
  - **核心语法**：
    ```sql
    SELECT column1, COUNT(*) FROM table_name GROUP BY column1;
    ```
  - **示例**：
    ```sql
    SELECT department, COUNT(*) FROM employees GROUP BY department;
    ```

- **HAVING**：
  - **定义**：用于对聚合后的结果进行过滤。
  - **核心语法**：
    ```sql
    SELECT column1, COUNT(*) FROM table_name GROUP BY column1 HAVING COUNT(*) > 10;
    ```
  - **示例**：
    ```sql
    SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 5;
    ```

- **ORDER BY**：
  - **定义**：用于对结果集进行排序（升序或降序）。
  - **核心语法**：
    ```sql
    SELECT column1, column2 FROM table_name ORDER BY column1 ASC/DESC;
    ```
  - **示例**：
    ```sql
    SELECT name, age FROM users ORDER BY age DESC;
    ```

**🔥 面试高频题**
1. 如何使用 `GROUP BY` 和 `HAVING` 进行分组和过滤？
   - **一句话答案**：`GROUP BY` 用于分组，`HAVING` 用于对分组后的结果进行过滤。
   - **深入回答**：
     - **GROUP BY**：将数据按指定列分组，常与聚合函数（如 `COUNT`、`SUM`）结合使用。
     - **HAVING**：对分组后的结果进行条件过滤，类似于 `WHERE`，但作用于聚合后的结果。
     ```sql
     SELECT department, COUNT(*) AS employee_count
     FROM employees
     GROUP BY department
     HAVING COUNT(*) > 5;
     ```

2. `ORDER BY` 的默认排序方式是什么？如何指定排序方向？
   - **一句话答案**：`ORDER BY` 默认按升序排序，通过 `ASC` 或 `DESC` 指定排序方向。
   - **深入回答**：
     - 默认情况下，`ORDER BY` 按升序（`ASC`）排序。
     - 使用 `DESC` 可以指定降序排序。
     ```sql
     SELECT name, age FROM users ORDER BY age DESC;
     ```

**🌟 重点提醒**
- **要点一**：`GROUP BY` 用于分组，`HAVING` 用于过滤分组后的结果。
- **要点二**：`ORDER BY` 默认升序，可通过 `ASC` 或 `DESC` 指定排序方向。

---

#### **2. 多表联接：INNER JOIN、LEFT JOIN、RIGHT JOIN、FULL JOIN**
**🔑 知识点详解**
- **INNER JOIN**：
  - **定义**：返回两个表中匹配的记录。
  - **核心语法**：
    ```sql
    SELECT columns FROM table1 INNER JOIN table2 ON table1.key = table2.key;
    ```
  - **示例**：
    ```sql
    SELECT users.name, orders.order_id
    FROM users INNER JOIN orders ON users.id = orders.user_id;
    ```

- **LEFT JOIN**：
  - **定义**：返回左表的所有记录，以及右表中匹配的记录；若无匹配，则右表字段为 NULL。
  - **核心语法**：
    ```sql
    SELECT columns FROM table1 LEFT JOIN table2 ON table1.key = table2.key;
    ```
  - **示例**：
    ```sql
    SELECT users.name, orders.order_id
    FROM users LEFT JOIN orders ON users.id = orders.user_id;
    ```

- **RIGHT JOIN**：
  - **定义**：返回右表的所有记录，以及左表中匹配的记录；若无匹配，则左表字段为 NULL。
  - **核心语法**：
    ```sql
    SELECT columns FROM table1 RIGHT JOIN table2 ON table1.key = table2.key;
    ```
  - **示例**：
    ```sql
    SELECT users.name, orders.order_id
    FROM users RIGHT JOIN orders ON users.id = orders.user_id;
    ```

- **FULL JOIN**：
  - **定义**：返回两个表中的所有记录，包括匹配和不匹配的记录。
  - **核心语法**：
    ```sql
    SELECT columns FROM table1 FULL JOIN table2 ON table1.key = table2.key;
    ```
  - **示例**：
    ```sql
    SELECT users.name, orders.order_id
    FROM users FULL JOIN orders ON users.id = orders.user_id;
    ```

**🔥 面试高频题**
1. `LEFT JOIN` 和 `INNER JOIN` 的区别是什么？
   - **一句话答案**：`INNER JOIN` 返回匹配的记录，`LEFT JOIN` 返回左表所有记录及右表匹配记录。
   - **深入回答**：
     - **INNER JOIN**：仅返回两个表中满足连接条件的记录。
     - **LEFT JOIN**：返回左表的所有记录，右表中无匹配时填充 NULL。
     ```sql
     -- INNER JOIN 示例
     SELECT users.name, orders.order_id
     FROM users INNER JOIN orders ON users.id = orders.user_id;
     
     -- LEFT JOIN 示例
     SELECT users.name, orders.order_id
     FROM users LEFT JOIN orders ON users.id = orders.user_id;
     ```

2. `FULL JOIN` 的应用场景是什么？
   - **一句话答案**：`FULL JOIN` 用于需要同时查看两个表中所有记录的场景。
   - **深入回答**：
     - **应用场景**：当需要分析两个表中所有记录的关系时，例如比较用户和订单数据。
     ```sql
     SELECT users.name, orders.order_id
     FROM users FULL JOIN orders ON users.id = orders.user_id;
     ```

**🌟 重点提醒**
- **要点一**：`INNER JOIN` 返回匹配记录，`LEFT JOIN` 返回左表所有记录。
- **要点二**：`FULL JOIN` 返回两个表中的所有记录。

---

#### **3. 子查询：相关子查询、非相关子查询**
**🔑 知识点详解**
- **非相关子查询**：
  - **定义**：子查询独立于外部查询，先执行子查询再执行外部查询。
  - **核心语法**：
    ```sql
    SELECT column1 FROM table1 WHERE column1 = (SELECT column2 FROM table2 WHERE condition);
    ```
  - **示例**：
    ```sql
    SELECT name FROM users WHERE age = (SELECT MAX(age) FROM users);
    ```

- **相关子查询**：
  - **定义**：子查询依赖于外部查询的每一行，逐行执行。
  - **核心语法**：
    ```sql
    SELECT column1 FROM table1 WHERE EXISTS (SELECT 1 FROM table2 WHERE table2.key = table1.key);
    ```
  - **示例**：
    ```sql
    SELECT name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
    ```

**🔥 面试高频题**
1. 相关子查询和非相关子查询的区别是什么？
   - **一句话答案**：非相关子查询独立于外部查询，相关子查询依赖于外部查询的每一行。
   - **深入回答**：
     - **非相关子查询**：子查询独立执行，结果传递给外部查询。
     - **相关子查询**：子查询逐行执行，依赖于外部查询的当前行。
     ```sql
     -- 非相关子查询
     SELECT name FROM users WHERE age = (SELECT MAX(age) FROM users);
     
     -- 相关子查询
     SELECT name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
     ```

**🌟 重点提醒**
- **要点一**：非相关子查询独立执行，相关子查询逐行执行。
- **要点二**：相关子查询通常用于复杂条件判断。

---

#### **4. 窗口函数：ROW_NUMBER、RANK、DENSE_RANK、LAG/LEAD**
**🔑 知识点详解**
- **ROW_NUMBER**：
  - **定义**：为结果集中的每一行分配唯一的行号。
  - **核心语法**：
    ```sql
    ROW_NUMBER() OVER (PARTITION BY column1 ORDER BY column2);
    ```
  - **示例**：
    ```sql
    SELECT name, age, ROW_NUMBER() OVER (ORDER BY age DESC) AS row_num FROM users;
    ```

- **RANK**：
  - **定义**：为结果集中的每一行分配排名，相同值共享排名并跳过后续排名。
  - **核心语法**：
    ```sql
    RANK() OVER (PARTITION BY column1 ORDER BY column2);
    ```
  - **示例**：
    ```sql
    SELECT name, age, RANK() OVER (ORDER BY age DESC) AS rank FROM users;
    ```

- **DENSE_RANK**：
  - **定义**：为结果集中的每一行分配排名，相同值共享排名但不跳过后续排名。
  - **核心语法**：
    ```sql
    DENSE_RANK() OVER (PARTITION BY column1 ORDER BY column2);
    ```
  - **示例**：
    ```sql
    SELECT name, age, DENSE_RANK() OVER (ORDER BY age DESC) AS dense_rank FROM users;
    ```

- **LAG/LEAD**：
  - **定义**：访问当前行前后相邻行的数据。
  - **核心语法**：
    ```sql
    LAG(column, offset) OVER (PARTITION BY column1 ORDER BY column2);
    LEAD(column, offset) OVER (PARTITION BY column1 ORDER BY column2);
    ```
  - **示例**：
    ```sql
    SELECT name, age, LAG(age, 1) OVER (ORDER BY age) AS prev_age FROM users;
    SELECT name, age, LEAD(age, 1) OVER (ORDER BY age) AS next_age FROM users;
    ```

**🔥 面试高频题**
1. `RANK` 和 `DENSE_RANK` 的区别是什么？
   - **一句话答案**：`RANK` 跳过后续排名，`DENSE_RANK` 不跳过。
   - **深入回答**：
     - **RANK**：相同值共享排名，后续排名跳过。
     - **DENSE_RANK**：相同值共享排名，后续排名不跳过。
     ```sql
     -- RANK 示例
     SELECT name, age, RANK() OVER (ORDER BY age DESC) AS rank FROM users;
     
     -- DENSE_RANK 示例
     SELECT name, age, DENSE_RANK() OVER (ORDER BY age DESC) AS dense_rank FROM users;
     ```

2. `LAG` 和 `LEAD` 的应用场景是什么？
   - **一句话答案**：`LAG` 用于访问前一行数据，`LEAD` 用于访问后一行数据。
   - **深入回答**：
     - **LAG**：常用于计算当前行与前一行的差异。
     - **LEAD**：常用于预测或分析未来趋势。
     ```sql
     -- LAG 示例
     SELECT name, age, LAG(age, 1) OVER (ORDER BY age) AS prev_age FROM users;
     
     -- LEAD 示例
     SELECT name, age, LEAD(age, 1) OVER (ORDER BY age) AS next_age FROM users;
     ```

**🌟 重点提醒**
- **要点一**：`RANK` 跳过后续排名，`DENSE_RANK` 不跳过。
- **要点二**：`LAG` 访问前一行数据，`LEAD` 访问后一行数据。

---

#### **5. 索引与优化：索引类型、执行计划、性能优化**
**🔑 知识点详解**
- **索引类型**：
  - **B-Tree 索引**：适用于范围查询和等值查询。
  - **哈希索引**：适用于等值查询，不支持范围查询。
  - **全文索引**：适用于文本搜索。

- **执行计划**：
  - **定义**：SQL 查询的执行路径，可通过 `EXPLAIN` 查看。
  - **核心语法**：
    ```sql
    EXPLAIN SELECT * FROM table_name WHERE condition;
    ```

- **性能优化**：
  - **索引优化**：为常用查询字段创建索引。
  - **查询优化**：避免全表扫描，减少子查询嵌套。
  - **分区表**：对大表进行分区，提升查询效率。

**🔥 面试高频题**
1. 如何通过索引优化查询性能？
   - **一句话答案**：为常用查询字段创建索引，避免全表扫描。
   - **深入回答**：
     - **创建索引**：为 `WHERE`、`JOIN` 和 `ORDER BY` 中的字段创建索引。
     - **避免冗余索引**：过多索引会增加写操作开销。
     ```sql
     CREATE INDEX idx_age ON users(age);
     ```

2. 如何分析 SQL 查询的执行计划？
   - **一句话答案**：使用 `EXPLAIN` 查看查询的执行路径，识别性能瓶颈。
   - **深入回答**：
     - **EXPLAIN 输出**：包含表扫描方式、索引使用情况等信息。
     - **优化建议**：优先使用索引扫描，避免全表扫描。
     ```sql
     EXPLAIN SELECT * FROM users WHERE age > 30;
     ```

**🌟 重点提醒**
- **要点一**：索引优化可显著提升查询性能。
- **要点二**：`EXPLAIN` 是分析查询性能的重要工具。

---

#### **6. 事务：ACID特性、隔离级别**
**🔑 知识点详解**
- **ACID 特性**：
  - **原子性（Atomicity）**：事务要么全部成功，要么全部失败。
  - **一致性（Consistency）**：事务完成后，数据保持一致状态。
  - **隔离性（Isolation）**：事务之间相互隔离，互不影响。
  - **持久性（Durability）**：事务提交后，数据永久保存。

- **隔离级别**：
  - **READ UNCOMMITTED**：最低隔离级别，允许脏读。
  - **READ COMMITTED**：防止脏读，但可能出现不可重复读。
  - **REPEATABLE READ**：防止脏读和不可重复读，但可能出现幻读。
  - **SERIALIZABLE**：最高隔离级别，完全隔离。

**🔥 面试高频题**
1. 数据库事务的 ACID 特性是什么？
   - **一句话答案**：ACID 包括原子性、一致性、隔离性和持久性。
   - **深入回答**：
     - **原子性**：事务要么全部成功，要么全部失败。
     - **一致性**：事务完成后，数据保持一致状态。
     - **隔离性**：事务之间相互隔离，互不影响。
     - **持久性**：事务提交后，数据永久保存。

2. 数据库的隔离级别有哪些？如何选择？
   - **一句话答案**：隔离级别包括 READ UNCOMMITTED、READ COMMITTED、REPEATABLE READ 和 SERIALIZABLE，根据业务需求选择。
   - **深入回答**：
     - **READ UNCOMMITTED**：允许脏读，性能最高。
     - **READ COMMITTED**：防止脏读，适合大多数场景。
     - **REPEATABLE READ**：防止脏读和不可重复读，适合高一致性需求。
     - **SERIALIZABLE**：完全隔离，性能最低。

**🌟 重点提醒**
- **要点一**：ACID 特性确保事务的可靠性。
- **要点二**：隔离级别影响并发性能和数据一致性。

---

#### **7. 数据操作：INSERT、UPDATE、DELETE**
**🔑 知识点详解**
- **INSERT**：
  - **定义**：向表中插入新记录。
  - **核心语法**：
    ```sql
    INSERT INTO table_name (column1, column2) VALUES (value1, value2);
    ```

- **UPDATE**：
  - **定义**：更新表中现有记录。
  - **核心语法**：
    ```sql
    UPDATE table_name SET column1 = value1 WHERE condition;
    ```

- **DELETE**：
  - **定义**：删除表中记录。
  - **核心语法**：
    ```sql
    DELETE FROM table_name WHERE condition;
    ```

**🔥 面试高频题**
1. 如何安全地执行批量删除操作？
   - **一句话答案**：在删除前使用 `SELECT` 确认删除条件，避免误删数据。
   
      - **深入回答**：
        
        - **安全步骤**：
          1. **确认条件**：使用 `SELECT` 查询验证删除条件是否正确。
          2. **备份数据**：在执行删除操作前，备份相关数据。
          3. **分步删除**：对于大批量数据，建议分批次删除以降低风险。
        ```sql
        -- 确认删除条件
        SELECT * FROM users WHERE age > 60;
        
        -- 执行删除操作
        DELETE FROM users WHERE age > 60;
        ```
   

2. 如何高效地插入大量数据？
   - **一句话答案**：使用批量插入或事务提交提升性能。
   - **深入回答**：
     - **批量插入**：将多条记录一次性插入，减少网络开销。
     - **事务控制**：将插入操作包裹在事务中，提升效率。
     ```sql
     -- 批量插入示例
     INSERT INTO users (name, age) VALUES ('Alice', 25), ('Bob', 30), ('Charlie', 35);
     
     -- 使用事务
     BEGIN TRANSACTION;
     INSERT INTO users (name, age) VALUES ('Alice', 25);
     INSERT INTO users (name, age) VALUES ('Bob', 30);
     COMMIT;
     ```

**🌟 重点提醒**
- **要点一**：在删除操作前使用 `SELECT` 验证条件，避免误删。
- **要点二**：批量插入和事务提交可显著提升插入性能。

---

#### **8. 视图：创建、更新、物化视图**
**🔑 知识点详解**
- **普通视图**：
  - **定义**：基于 SQL 查询的虚拟表，不存储实际数据。
  - **核心语法**：
    ```sql
    CREATE VIEW view_name AS SELECT column1, column2 FROM table_name WHERE condition;
    ```
  - **示例**：
    ```sql
    CREATE VIEW active_users AS SELECT name, age FROM users WHERE status = 'active';
    ```

- **更新视图**：
  - **定义**：通过视图更新基础表的数据（需满足一定条件）。
  - **核心语法**：
    ```sql
    UPDATE view_name SET column1 = value1 WHERE condition;
    ```
  - **示例**：
    ```sql
    UPDATE active_users SET age = 26 WHERE name = 'Alice';
    ```

- **物化视图**：
  - **定义**：存储查询结果的实际数据，定期刷新。
  - **核心语法**：
    ```sql
    CREATE MATERIALIZED VIEW mv_name AS SELECT column1, column2 FROM table_name WHERE condition;
    ```
  - **示例**：
    ```sql
    CREATE MATERIALIZED VIEW mv_active_users AS SELECT name, age FROM users WHERE status = 'active';
    ```

**🔥 面试高频题**
1. 普通视图和物化视图的区别是什么？
   - **一句话答案**：普通视图是虚拟表，不存储数据；物化视图存储实际数据，需定期刷新。
   - **深入回答**：
     - **普通视图**：基于 SQL 查询动态生成结果，实时性高但性能较低。
     - **物化视图**：存储查询结果，性能高但需定期刷新以保持数据一致性。
     ```sql
     -- 普通视图
     CREATE VIEW active_users AS SELECT name, age FROM users WHERE status = 'active';
     
     -- 物化视图
     CREATE MATERIALIZED VIEW mv_active_users AS SELECT name, age FROM users WHERE status = 'active';
     ```

2. 如何刷新物化视图？
   - **一句话答案**：使用 `REFRESH MATERIALIZED VIEW` 命令刷新物化视图。
   - **深入回答**：
     - **手动刷新**：显式调用刷新命令。
     - **自动刷新**：配置定时任务定期刷新。
     ```sql
     REFRESH MATERIALIZED VIEW mv_active_users;
     ```

**🌟 重点提醒**
- **要点一**：普通视图不存储数据，物化视图存储数据。
- **要点二**：物化视图需定期刷新以保持数据一致性。

---

#### **💡 复习建议**
1. 掌握基础查询语法（`SELECT`、`WHERE`、`GROUP BY`、`HAVING`、`ORDER BY`）及其应用场景。
2. 理解多表联接（`INNER JOIN`、`LEFT JOIN`、`RIGHT JOIN`、`FULL JOIN`）的区别及使用场景。
3. 学习子查询（相关子查询、非相关子查询）的实现方式。
4. 掌握窗口函数（`ROW_NUMBER`、`RANK`、`DENSE_RANK`、`LAG/LEAD`）的核心用法。
5. 理解索引类型、执行计划分析及性能优化方法。
6. 掌握事务的 ACID 特性和隔离级别。
7. 熟悉数据操作（`INSERT`、`UPDATE`、`DELETE`）的最佳实践。
8. 理解普通视图和物化视图的区别及其适用场景。


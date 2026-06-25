# RAG Hetero-Parallel Hang Debug Note

Date: 2026-03-30

## Summary

`hetero_parallel` 在 Android 设备上会随机卡住。最终确认这不是上层 RAG 流程锁死，也不只是 QNN forward 重叠，而是 GGML 线程池在销毁阶段存在竞态，导致 `reset_threadpool()` 偶发永久等待。

修复后，设备侧行为恢复稳定，不再复现此前的中途 hang。

## Symptom

- 请求模式：`/v1/rag` + `mode=hetero_parallel`
- 现象：server 不崩溃，但请求卡在中途，后续无响应
- 早期观察：关闭全局串行后更容易复现，说明问题和并发有关

一次关键 hang 日志停在这里：

- `qwen3 embedding threadpool reset begin`
- `ggml threadpool reset begin`
- 没有对应的 reset end

这说明卡点已经不在 QNN forward 主体，而是在 embedding 收尾阶段的线程池回收。

## Evidence Chain

### 1. 设备侧不是简单互斥死锁

Android `/proc/<pid>/task/*` 快照显示：

- 主线程在 `inet_csk_accept`
- 一部分线程在 `fastrpc_wait_for_completion` / `fastrpc_device_ioctl`
- 大量线程在 `futex_wait_queue`

这说明运行时确实存在后端等待，但应用层也有线程池/barrier 等待链条。

### 2. QNN forward 串行化不能根治

即使设置 `POWERSERVE_QNN_FORWARD_SERIAL=1`，hang 仍然出现。

因此根因不能只解释为“两个 QNN forward 重叠进入后端”。

### 3. 关键挂点落在 threadpool reset

通过阶段日志，hang 案例中：

- `doc_embedding.chunk_2` 已完成 executor 计算
- embedding 抽取阶段已完成
- 卡在 `ggml threadpool reset begin`

这把问题范围缩小到 `GGMLBackend::reset_threadpool()` 和其内部 `ThreadPool` 析构。

## Root Cause

`ThreadPool::~ThreadPool()` 通过 `m_run_barrier` 唤醒 worker，再 `join()` 各线程。

但旧的 `ThreadPool::thread_main()` 逻辑是：

1. 在循环顶部先检查 `!m_exited`
2. 再进入 `m_run_barrier`

这会产生竞态窗口：

- 析构线程把 `m_exited` 置为 `true`
- 某些 worker 还没进入 barrier，就先看到 `m_exited == true`
- 这些 worker 直接退出循环，不再参与这次 barrier
- 析构线程永久卡在 `spin_barrier_wait(&m_run_barrier)`

也就是：销毁阶段需要的 barrier 参与者数不再满足，导致偶发永久挂死。

## Fix

保留的修复在：

- `src/core/thread_pool.cpp`

修复思路：

- worker 必须先进入 `m_run_barrier`
- barrier 返回后再检查 `m_exited`

这样析构阶段唤醒用的那一次 barrier 一定能凑齐参与者，线程才能安全退出，不会把销毁线程遗留在 barrier 中。

## Verification

### Host-side regression probe

新增最小回归探针：

- `tests/thread_pool_shutdown_race.cpp`
- `tests/check_thread_pool_shutdown_race.sh`

用途：

- 反复创建 `ThreadPool`
- 跑一轮小任务
- 立即销毁

这能专门打到“线程池销毁竞态”而不依赖完整 RAG/QNN 环境。

本次验证命令：

```bash
bash tests/check_thread_pool_shutdown_race.sh
```

验证结果：

- `thread_pool_shutdown_race: ok (32 iterations)`
- exit code `0`

### Device-side result

应用修复后，Android 设备上的 `hetero_parallel` 请求已恢复稳定，原先的随机 hang 不再复现。

## Final Keep/Drop Decision

保留：

- `src/core/thread_pool.cpp` 中的销毁竞态修复
- `tests/thread_pool_shutdown_race.cpp`
- `tests/check_thread_pool_shutdown_race.sh`
- 本文档

删除：

- 本轮为定位问题临时加入的 execution tracing 框架
- QNN/GGML/executor/qwen3 的阶段性调试日志
- Android 线程/栈采集脚本

这样最终补丁只保留真正有长期价值的修复和最小回归验证。

# Kubernetes 集群运维手册

> **适用范围**: KnowFlow 生产环境
> **版本**: v1.3
> **维护者**: 基础架构组

---

## 1. 集群架构

### 节点列表

| 节点名称 | IP | 角色 | 状态 |
|---------|-----|------|------|
| kf-master-01 | 10.0.1.10 | Control Plane | Running |
| kf-worker-01 | 10.0.1.21 | Backend | Running |
| kf-worker-02 | 10.0.1.22 | Backend | Running |
| kf-worker-03 | 10.0.1.23 | Milvus | Running |
| kf-worker-04 | 10.0.1.24 | PostgreSQL | Running |

---

## 2. 日常巡检

### 2.1 检查集群健康状态

```bash
kubectl get nodes
kubectl get pods -n knowflow
kubectl get svc -n knowflow
kubectl get pvc -n knowflow
```

### 2.2 检查资源使用

```bash
kubectl top nodes
kubectl top pods -n knowflow --sort-by=cpu
```

---

## 3. 常见故障处理

### 3.1 Pod 重启循环 (CrashLoopBackOff)

**排查步骤**:

1. 查看 Pod 事件: `kubectl describe pod <pod-name> -n knowflow`
2. 查看 Error 日志: `kubectl logs <pod-name> -n knowflow | grep -i error | tail -20`
3. 常见原因:
   - PostgreSQL / Milvus 连接失败 → 检查 Service 和 Endpoint
   - OOM → 调整 resources.limits.memory
   - 配置文件错误 → 检查 ConfigMap

### 3.2 Milvus 检索延迟升高

1. 检查 Index 状态: `curl http://localhost:9091/healthz`
2. 检查磁盘 IO: `iostat -x 1 5`
3. 临时措施：触发索引重建

### 3.3 PostgreSQL 主从延迟

```sql
SELECT * FROM pg_stat_replication;
SELECT application_name, state, sync_state,
  pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag,
  pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn) AS write_lag
FROM pg_stat_replication;
```

---

## 4. 备份与恢复

### 数据库备份

```bash
kubectl exec -it deployment/knowflow-postgres -n knowflow -- \
  pg_dump -U knowflow -Fc knowflow > knowflow_$(date +%Y%m%d).dump
```

### 恢复流程

```bash
# 1. 停止写入
kubectl scale deployment knowflow-backend -n knowflow --replicas=0
# 2. 恢复数据库
kubectl exec -it deployment/knowflow-postgres -n knowflow -- \
  pg_restore -U knowflow -d knowflow --clean knowflow_20260623.dump
# 3. 重建索引
# 4. 恢复服务
kubectl scale deployment knowflow-backend -n knowflow --replicas=3
```

---

## 5. 扩容指南

```bash
# Backend 水平扩容 (HPA)
kubectl autoscale deployment knowflow-backend -n knowflow \
  --cpu-percent=70 --min=3 --max=20

# Milvus DataNode 扩容
kubectl scale deployment knowflow-milvus-datanode -n knowflow --replicas=5
```

> **最后更新**: 2026-06-23
> **下次审查**: 2026-09-23

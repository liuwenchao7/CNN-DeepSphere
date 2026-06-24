# 2026-06-24 GitHub 上传准备

## 目标

将项目推送至 https://github.com/liuwenchao7/CNN-DeepSphere ，文档格式参考 [CNN-PointNet](https://github.com/mrheng9/CNN-PointNet)。

## 已完成

1. **初始化 git 仓库**（`/disk_pool1/liuwc/cursor_workspace/pid`）
2. **新增 `.gitignore`**：排除 `logs/`、checkpoint、`.npy`/`.npz`；保留 `outputs/` 下 json、manifest、loss_log 等元数据
3. **新增 `docs/`**：汇总 `EXECUTION_PLAN.md`、`FULL_EXPERIMENT_STATUS.md`、`DATA_AND_VISUALIZATION.md`、`SCORE_AND_BOOTSTRAP.md`、`PAPER_PID_NOTES.md`
4. **新增 `readme.md`**：英文教程式说明（Repository Layout / Method / Data / Environment / Training / FAQ），对齐 CNN-PointNet 结构
5. **更新 `README.md`**：顶部增加指向 `readme.md` 的链接
6. **首次提交**：189 个文件（代码、文档、audit 图、实验元数据）
7. **合并远端 stub README** 并生成本地 bundle 备份：`CNN-DeepSphere-main.bundle`（5.1 MB）

## 未推送原因

提供的 GitHub PAT 对该仓库 **无写入权限**（`git push` 与 Contents API 均返回 403 / `Resource not accessible by personal access token`）。

## 用户需完成的推送步骤

1. 在 GitHub → Settings → Developer settings 生成 **Fine-grained PAT** 或 **Classic PAT（repo 权限）**，并勾选仓库 `liuwenchao7/CNN-DeepSphere` 的 **Contents: Read and write**
2. **立即撤销** 本次对话中暴露的旧 token
3. 推送（任选其一）：

```bash
cd /disk_pool1/liuwc/cursor_workspace/pid
git push https://liuwenchao7:<NEW_TOKEN>@github.com/liuwenchao7/CNN-DeepSphere.git main
```

或从 bundle 恢复后推送：

```bash
git clone CNN-DeepSphere-main.bundle CNN-DeepSphere
cd CNN-DeepSphere
git remote add origin https://github.com/liuwenchao7/CNN-DeepSphere.git
git push -u origin main
```

## 未纳入 git 的内容（按设计）

- 原始 NPZ 数据（`/disk_pool1/weijsh/...` 等集群路径）
- WFSampling 预处理输出（`WFS_wav_v2/` 等）
- 训练 checkpoint 权重、`logs/` 大日志

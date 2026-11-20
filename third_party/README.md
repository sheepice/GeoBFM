# 固定依赖源码

为保留已有 SONIC/ViBe 检查点接口，本目录使用现有实验实际固定的依赖版本，而不是安装名称相同但接口不同的任意最新版 PyPI 包。

目录内只有选取的 Python 源码、小配置、包元数据、README 和许可证。没有子仓库 `.git`、权重、动作集、机器人 mesh、视频或环境缓存。它们作为普通文件随本仓库提交，因此远程 clone 不依赖私有子模块地址。

版本、来源和文件哈希见 [manifest.json](manifest.json)，完整许可索引见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。`deps.lock` 保留上游源信息，并用于上游根目录定位；本仓库安装路径以 `scripts/setup_env.sh` 为准。不要在这里运行全量同步 / 数据下载流程。

原上游 README 可能引用此次未复制的完整项目脚本与文档，这是源码快照而非上游仓库镜像。新项目的安装和训练说明以根 README 为准。新增功能优先写在 `training/` 或 `src/`；确需改第三方实现时保留改动说明并更新校验记录。

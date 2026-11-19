# paper-code-coach · 论文代码训练教练

把论文里的公式、算法和代码入口串成可以动手验证的实现练习。这是一个面向 Codex 的技能，默认用中文讲解，以 Python 为练习语言。神经网络练习没有指定框架时，默认使用 PyTorch。

## 安装

在 PowerShell、Bash 或 Zsh 中运行以下命令，将仓库克隆到 Codex 的用户技能目录。克隆时需要有仓库访问权限。

```sh
git clone https://github.com/4Lily/paper-code-coach.git "$HOME/.agents/skills/paper-code-coach"
```

只在某个项目里使用时，可以把克隆目标改为该项目根目录下的 `.agents/skills/paper-code-coach`。已有同名技能时，请更新原副本或选择一份使用。若技能列表尚未刷新，重新打开 Codex。 [Codex 技能文档](https://developers.openai.com/codex/skills)

## 开始训练

从练习项目中新建 Codex 对话，提供论文 PDF 或链接、参考仓库与版本，以及你希望完成的模块。用 `$paper-code-coach` 显式调用技能。

> 用 $paper-code-coach 带我复现这篇论文。先对照我给出的论文和参考仓库，找出第一个能独立验证的模块，并在练习目录里写出这一模块的讲义。目标函数由我实现。

> 用 $paper-code-coach 验收我写好的模块。先从正常调用入口跑一个最小样例，把预期值和实际结果列出来，再指出我应该回查哪一步。

## 一次模块训练怎么进行

1. **对齐论文与实现。** 确认论文版本、源码入口和实际使用的配置，定位公式、算法或接口。
2. **在项目里写讲义。** 每个完整模块包含「原理与作用」「函数用法」「实现步骤」。同一组小输入贯穿手算、API 调用和自查。优先沿用项目讲义，首次创建时默认命名为 `教学.md`。
3. **由学员完成实现。** 卡住时可围绕当前尝试逐步补提示。明确提出时，教练也可以提供完整实现或修复。
4. **从实际入口验收。** 用能区分正确与错误行为的输入核对输出，注明是手算预期、静态检查还是运行结果。续学时可从已确认的进度和卡点接着做。

练习源码、学习档案和实验报告由使用者保存在自己选定的位置。仓库内的示例是技能自带的讲解材料。 [点云中心化示例](references/examples.md#点云中心化短讲义)

## 仓库内容

| 文件 | 用途 |
| --- | --- |
| [`SKILL.md`](SKILL.md) | 技能入口，描述触发场景和训练流程 |
| [`agents/openai.yaml`](agents/openai.yaml) | Codex 界面名称、简介和默认提示词 |
| [`references/teaching.md`](references/teaching.md) | 模块拆分与三节讲义的写法 |
| [`references/sources.md`](references/sources.md) | 论文版本、公式与实际代码入口的核对方法 |
| [`references/review.md`](references/review.md) | 最小验证用例的选择与验收反馈 |
| [`references/learning-profile.md`](references/learning-profile.md) | 学习记录的维护与跨会话衔接 |
| [`references/examples.md`](references/examples.md) | 可用于备课的短讲义和 API 示例 |

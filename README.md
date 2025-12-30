# ComfyUI-PainterQwenImageEdit 本节点由抖音博主：绘画小子 制作

**Qwen多图编辑增强节点** | 支持8图输入 · 双条件输出 · 背景层优先

---

## 功能特点

### 🎨 **八图协同编辑**
- **1张背景图** + **7张叠加图**的完整工作流程
- 背景图自动置于token序列首位，确保合成基准
- 所有图像输入均为**可选**，灵活适配不同场景

### ⚡ **双条件输出**
- **正面条件**：正常处理您的prompt和图像输入
- **负面条件**：自动生成零化(empty)条件，无需手动创建空提示词
- 输出端口清晰标注为"正面条件"和"负面条件"，直连KSampler使用

### 🎯 **原生功能完整保留**
- 完整支持Qwen的llama_template系统提示词
- 自动reference_latents编码（连接VAE时）
- 智能图像缩放（384px预览 + 1024px潜空间编码）
- 动态提示词(Dynamic Prompts)支持

---

## 安装方法

### 通过ComfyUI-Manager（推荐）
在Manager中搜索 `PainterQwenImageEdit` 直接安装

### 手动安装
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/princepainter/ComfyUI-PainterQwenImageEdit.git
```
安装后**重启ComfyUI**即可

---

## 节点参数说明

### 输入参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `clip` | CLIP | 必需，CLIP模型 |
| `prompt` | STRING | 必需，编辑指令（支持多行和动态提示词） |
| `vae` | VAE | 可选，用于生成reference_latents |
| `background` | IMAGE | **可选**，背景图像（优先级最高） |
| `image1` - `image7` | IMAGE | 可选，叠加图像1-7 |

### 输出参数
| 参数 | 类型 | 说明 |
|------|------|------|
| `正面条件` | CONDITIONING | 处理后的正向条件，连至KSampler的`positive` |
| `负面条件` | CONDITIONING | 零化负向条件，连至KSampler的`negative` |

---

## 使用示例

**典型工作流：**
1. 加载背景图 → `background` 输入端
2. 加载前景元素 → `image1`, `image2` 等输入端
3. 输入编辑指令 → `prompt`（如："在背景上添加前景物体，保持风格一致"）
4. 连接CLIP和VAE
5. 输出直接连至 **KSampler** 的 `positive` 和 `negative` 端口

---

## 版本兼容

- **ComfyUI**: v0.6.0+ (支持传统节点格式)
- **Python**: 3.8+
- **框架**: 原生兼容ComfyUI的Qwen模型工作流

---


---

**问题反馈**：欢迎提交Issue或PR，好用请给我点个星星，多谢！🙏

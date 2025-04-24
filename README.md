Project_Code
本项目旨在提供一个完整的机器学习分类与预测框架，涵盖数据预处理、特征选择、模型训练与评估等模块。

📁 项目结构
Project_Code/
├── classification/                 # 分类模型代码
├── classification_project/         # 分类项目主目录
├── classification_project_stability/ # 稳定性分析相关代码
├── dataset/                        # 数据集文件夹
├── iRFDA-master/                   # 引用的 iRFDA 算法实现
├── prediction/                     # 预测模型代码
├── .history/                       # 编辑器历史记录（可忽略）
├── .DS_Store                       # macOS 系统文件（可忽略）
🧰 环境依赖
本项目主要使用 Python 和 MATLAB 开发。​

Python 依赖
numpy
pandas
scikit-learn
matplotlib
seaborn

可以使用以下命令安装：​
pip install numpy pandas scikit-learn matplotlib seaborn

MATLAB 依赖
部分算法（如 iRFDA）在 MATLAB 环境下运行，确保已安装 MATLAB 及相关工具箱。​
🚀 快速开始
1. 克隆项目
git clone https://github.com/ZuoSQi/Project_Code.git
cd Project_Code
2. 数据准备
将您的数据集放置于 dataset/ 文件夹中，确保数据格式与代码要求一致。​
3. 运行分类模型
进入分类项目目录并运行主程序：​
cd classification_project
python main.py
4. 运行预测模型
进入预测模型目录并运行主程序：​
cd prediction
python predict.py

📊 功能模块
分类模型：​支持多种机器学习算法，包括支持向量机、随机森林等。​
预测模型：​实现了时间序列预测和回归分析。​
稳定性分析：​评估模型在不同参数和数据扰动下的稳定性。​
iRFDA 算法：​集成了改进的鲁棒判别分析算法，增强模型的泛化能力。

📈 示例结果
项目运行后将在 results/ 文件夹中生成以下内容：​
模型评估指标（准确率、召回率、F1 分数等）​
混淆矩阵图​
特征重要性分析图​
预测结果可视化图​

import pandas as pd
import numpy as np
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class KnowledgeFeatureExtractor:
    """知识点特征提取器"""
    
    def __init__(self, knowledge_hierarchy: Dict[str, List[str]] = None):
        """
        初始化特征提取器
        
        Args:
            knowledge_hierarchy: 知识点层次结构字典
                格式: {"大类1": ["子类1", "子类2"], "大类2": [...]}
        """
        self.knowledge_hierarchy = knowledge_hierarchy or {
            'algebra':['方程', '函数', '不等式', '多项式', '数列','同类项','公式','一元二次方程','配方','集合','一次函数', '代数式', '复合运算', '归一','一元一次','二元一次', '一元一次不等式','不等式','方程组','一元一次方程','移项','一元二次', '函数','变形', '多项式','应用题','分配问题', '关系', '一元一次不等式组','等式','二次函数','二元一次方程','代数', '二次方程','函数关系', '方程', '应用'],
            'geometry':['三角形', '圆形', '面积', '体积', '相似', '全等','等腰三角形','侧面','侧面积','内角', '三角形', '长方体','正方形', '圆环','特征','位置','二元一次方程组','面积','多边形', '长度', '四边形', '周长', '圆锥','平行四边形','底面', '正方体','平行', '梯形','移动','三角', '面积单位','圆柱', '几何', '长方形','表面','体积','外角','棱长', '全等三角形'],
            'arithmetic':['分数', '小数', '整数运算', '比例', '百分数','负数','平方','除法','比','余数','估算','倍数','平均数', '数轴', '小数点', '加法','因数','四则','百分数','完全平方',  '进位','最小公倍数', '质数', '公倍数','公因数', '积分','分配','减法', '最大公因数','四则运算','平均','计算', '近似数','整数','量','数字', '乘法','积和','数','最大','最小', '运算','自然数', '分数'],
            'statisticalProbability':['平均数', '概率', '统计图表', '可能性','相关','计量','比例','质量'],
            'commonKnowledge':['单位','距离','周期', '字母表','性质','选择','复合','长度单位','折扣','定义','字问题','问题', '完全','原理', '表示', '括号','方法','法则',]
        }
        
        # 构建知识点到大类的反向映射
        self.knowledge_to_category = {}
        for category, knowledge_list in self.knowledge_hierarchy.items():
            for knowledge in knowledge_list:
                self.knowledge_to_category[knowledge] = category
        
        # 存储共现网络
        self.knowledge_cooccurrence_network = None
        self.knowledge_centrality = None
    
    def parse_knowledge_string(self, knowledge_str: str) -> List[str]:
        """
        解析知识点字符串，提取知识点列表
        
        Args:
            knowledge_str: 知识点字符串，如"分数;比例;方程"
            
        Returns:
            知识点列表
        """
        if pd.isna(knowledge_str) or not knowledge_str:
            return []
        
        # 去除空格并按分隔符分割
        separators = [';', '；', ',', '，', '、', '|',' ']
        for sep in separators:
            if sep in knowledge_str:
                return [k.strip() for k in knowledge_str.split(sep) if k.strip()]
        
        # 如果没有分隔符，直接返回
        return [knowledge_str.strip()]
    
    def extract_basic_features(self, knowledge_list: List[str]) -> Dict:
        """
        提取基础特征
        
        Args:
            knowledge_list: 知识点列表
            
        Returns:
            基础特征字典
        """
        features = {}
        
        # 1. 知识点数量
        features['knowledge_count'] = len(knowledge_list)
        
        # 2. 知识类型分布
        category_count = Counter()
        for knowledge in knowledge_list:
            # 查找知识点所属大类
            category_found = False
            for category, subcategories in self.knowledge_hierarchy.items():
                # 检查知识点是否包含子类的关键词
                for subcat in subcategories:
                    if subcat in knowledge:
                        category_count[category] += 1
                        category_found = True
                        break
                if category_found:
                    break
            
            # 如果没找到，尝试直接匹配
            if not category_found and knowledge in self.knowledge_to_category:
                category_count[self.knowledge_to_category[knowledge]] += 1
            elif not category_found:
                category_count['others'] += 1
        
        # 将类别计数转换为特征
        for category in self.knowledge_hierarchy.keys():
            features[f'category_{category}'] = category_count.get(category, 0)
        features['category_others'] = category_count.get('others', 0)
        
        # 3. 知识类型多样性（使用香农熵）
        if knowledge_list:
            total = sum(category_count.values())
            entropy = 0
            for count in category_count.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            features['knowledge_entropy'] = entropy
        else:
            features['knowledge_entropy'] = 0
        
        # 4. 知识深度（平均层次深度）
        depth_scores = []
        for knowledge in knowledge_list:
            depth = 1  # 默认深度为1
            for category, subcategories in self.knowledge_hierarchy.items():
                for i, subcat in enumerate(subcategories, 1):
                    if subcat in knowledge:
                        depth = max(depth, i + 1)  # 类别为1，子类为2,3,...
            depth_scores.append(depth)
        
        features['avg_knowledge_depth'] = np.mean(depth_scores) if depth_scores else 0
        features['max_knowledge_depth'] = max(depth_scores) if depth_scores else 0
        
        return features
    
    def build_cooccurrence_network(self, all_knowledge_lists: List[List[str]]):
        """
        构建知识点共现网络
        
        Args:
            all_knowledge_lists: 所有题目的知识点列表
        """
        # 初始化图
        G = nx.Graph()
        
        # 统计共现
        for knowledge_list in all_knowledge_lists:
            if len(knowledge_list) >= 2:
                # 添加节点
                for knowledge in knowledge_list:
                    if not G.has_node(knowledge):
                        G.add_node(knowledge, count=0)
                    G.nodes[knowledge]['count'] += 1
                
                # 添加边（共现关系）
                for i, j in itertools.combinations(knowledge_list, 2):
                    if G.has_edge(i, j):
                        G[i][j]['weight'] += 1
                        G[i][j]['cooccur_count'] += 1
                    else:
                        G.add_edge(i, j, weight=1, cooccur_count=1)
        
        self.knowledge_cooccurrence_network = G
        
        # 计算中心性指标
        if len(G.nodes()) > 0:
            # 度中心性
            degree_centrality = nx.degree_centrality(G)
            # 中介中心性（仅对较小网络）
            if len(G.nodes()) < 2000:
                betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
            else:
                betweenness_centrality = {node: 0 for node in G.nodes()}
            
            self.knowledge_centrality = {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality
            }
    
    def extract_network_features(self, knowledge_list: List[str]) -> Dict:
        """
        提取网络相关特征
        
        Args:
            knowledge_list: 知识点列表
            
        Returns:
            网络特征字典
        """
        features = {}
        
        if not self.knowledge_cooccurrence_network or not knowledge_list:
            features.update({
                'knowledge_interaction_density': 0,
                'avg_degree_centrality': 0,
                'avg_betweenness_centrality': 0,
                'max_degree_centrality': 0,
                'clustering_coefficient': 0
            })
            return features
        
        G = self.knowledge_cooccurrence_network
        
        # 1. 知识交互密度（知识点间的平均连接强度）
        if len(knowledge_list) >= 2:
            total_weight = 0
            edge_count = 0
            for i, j in itertools.combinations(knowledge_list, 2):
                if G.has_edge(i, j):
                    total_weight += G[i][j]['weight']
                    edge_count += 1
            
            # 计算密度：实际连接强度与可能连接数之比
            possible_edges = len(knowledge_list) * (len(knowledge_list) - 1) / 2
            if possible_edges > 0:
                features['knowledge_interaction_density'] = total_weight / possible_edges
            else:
                features['knowledge_interaction_density'] = 0
        else:
            features['knowledge_interaction_density'] = 0
        
        # 2. 中心性特征
        centrality_scores = []
        betweenness_scores = []
        
        for knowledge in knowledge_list:
            if knowledge in self.knowledge_centrality['degree']:
                centrality_scores.append(self.knowledge_centrality['degree'][knowledge])
                betweenness_scores.append(self.knowledge_centrality['betweenness'][knowledge])
        
        features['avg_degree_centrality'] = np.mean(centrality_scores) if centrality_scores else 0
        features['avg_betweenness_centrality'] = np.mean(betweenness_scores) if betweenness_scores else 0
        features['max_degree_centrality'] = max(centrality_scores) if centrality_scores else 0
        
        # 3. 聚类系数（子图的紧密度）
        if len(knowledge_list) >= 3:
            subgraph = G.subgraph(knowledge_list)
            try:
                clustering = nx.average_clustering(subgraph)
                features['clustering_coefficient'] = clustering
            except:
                features['clustering_coefficient'] = 0
        else:
            features['clustering_coefficient'] = 0
        
        return features
    
    def extract_text_complexity_features(self, knowledge_list: List[str]) -> Dict:
        """
        提取文本复杂度特征
        
        Args:
            knowledge_list: 知识点列表
            
        Returns:
            文本特征字典
        """
        features = {}
        
        if not knowledge_list:
            features.update({
                'avg_knowledge_length': 0,
                'knowledge_specificity': 0
            })
            return features
        
        # 1. 平均知识点名称长度
        lengths = [len(knowledge) for knowledge in knowledge_list]
        features['avg_knowledge_length'] = np.mean(lengths)
        
        # 2. 知识点特异性（通过逆文档频率估算）
        if self.knowledge_cooccurrence_network:
            total_nodes = len(self.knowledge_cooccurrence_network.nodes())
            node_counts = {node: self.knowledge_cooccurrence_network.nodes[node]['count'] 
                          for node in self.knowledge_cooccurrence_network.nodes()}
            
            specificity_scores = []
            for knowledge in knowledge_list:
                if knowledge in node_counts:
                    # 使用log(总题目数/出现次数)作为特异性指标
                    if node_counts[knowledge] > 0:
                        idf_score = np.log(total_nodes / node_counts[knowledge])
                        # idf_score = total_nodes / node_counts[knowledge]
                        specificity_scores.append(idf_score)
            
            features['knowledge_specificity'] = np.mean(specificity_scores) if specificity_scores else 0
        else:
            features['knowledge_specificity'] = 0
        
        return features
    
    def extract_all_features(self, knowledge_list: List[str]) -> Dict:
        """
        提取所有特征
        
        Args:
            knowledge_list: 知识点列表
            
        Returns:
            所有特征的字典
        """
        # 提取各类特征
        features = {}
        
        # 基础特征
        basic_features = self.extract_basic_features(knowledge_list)
        features.update(basic_features)
        
        # 网络特征
        network_features = self.extract_network_features(knowledge_list)
        features.update(network_features)
        
        # 文本复杂度特征
        text_features = self.extract_text_complexity_features(knowledge_list)
        features.update(text_features)
        
        return features

    def visualize_knowledge_network(self, top_n=30, figsize=(12, 10)):
        """可视化知识点共现网络"""
        if not self.knowledge_cooccurrence_network:
            print("尚未构建知识网络，请先调用 build_cooccurrence_network 方法")
            return
        
        G = self.knowledge_cooccurrence_network
        
        # 选择前top_n个最重要的节点（按度中心性）
        if self.knowledge_centrality and len(G.nodes()) > top_n:
            top_nodes = sorted(
                self.knowledge_centrality['degree'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            top_node_names = [node for node, _ in top_nodes]
            H = G.subgraph(top_node_names)
        else:
            H = G
            top_n = min(top_n, len(G.nodes()))
        
        # 绘制网络
        plt.figure(figsize=figsize)
        
        # 计算节点大小（基于度中心性）
        if self.knowledge_centrality:
            node_sizes = [self.knowledge_centrality['degree'].get(node, 0.1) * 3000 + 100 
                         for node in H.nodes()]
        else:
            node_sizes = 300
        
        # 计算节点颜色（基于中介中心性）
        if self.knowledge_centrality:
            node_colors = [self.knowledge_centrality['betweenness'].get(node, 0) 
                          for node in H.nodes()]
        else:
            node_colors = 'lightblue'
        # 绘制网络
        pos = nx.spring_layout(H, k=2, iterations=50, seed=42)
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            H, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.plasma,
            alpha=0.8
        )
        
        # 绘制边
        edges = nx.draw_networkx_edges(
            H, pos,
            alpha=0.2,
            width=1
        )
        
        # 绘制标签
        labels = {node: node for node in H.nodes()}
        nx.draw_networkx_labels(
            H, pos, 
            labels=labels,
            font_size=10,
            font_family='sans-serif'
        )
        
        plt.title(f'知识点共现网络（前{top_n}个核心知识点）', fontsize=16, pad=20)
        plt.axis('off')
        
        # 添加颜色条
        if isinstance(node_colors, list):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                      norm=plt.Normalize(vmin=min(node_colors), 
                                                        vmax=max(node_colors)))
            sm._A = []
            cbar = plt.colorbar(sm, shrink=0.8)
            cbar.set_label('中介中心性', fontsize=12)
        
        plt.tight_layout()
        return plt.gcf()  
# features_df.to_csv('knowledge_features.csv', encoding='utf-8-sig')
# print("\n特征数据已保存到: knowledge_features.csv")
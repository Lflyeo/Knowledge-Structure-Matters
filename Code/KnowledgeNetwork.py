import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文
# plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class KnowledgeNetworkAnalyzer:
    """
    知识点网络分析器
    """
    
    def __init__(self, knowledge_graph):
        """
        初始化分析器
        
        Args:
            knowledge graph
        """
        self.G = knowledge_graph  # 网络图
        self.communities = None  # 社区划分
        self.hierarchy = None  # 层次结构
    
    def analyze_network_topology(self):
        """分析网络拓扑结构"""
        if self.G is None or self.G.number_of_nodes() == 0:
            print("请先构建网络")
            return None
        
        G = self.G
        
        # 1. 基本统计
        metrics = {
            '节点数': G.number_of_nodes(),
            '边数': G.number_of_edges(),
            '网络密度': nx.density(G),
            '平均度': np.mean([d for n, d in G.degree()]),
            '平均聚类系数': nx.average_clustering(G),
            '平均路径长度': nx.average_shortest_path_length(G) if nx.is_connected(G) else None,
            '直径': nx.diameter(G) if nx.is_connected(G) else None,
        }
        
        # 2. 度分布（幂律检验）
        degrees = [d for n, d in G.degree()]
        metrics.update({
            '最大度': max(degrees),
            '最小度': min(degrees),
            '度分布偏度': pd.Series(degrees).skew(),
            '度分布峰度': pd.Series(degrees).kurtosis(),
        })
        
        # 3. 中心性指标
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # 找出中心知识点
        top_k = 10
        metrics['度中心性前10'] = sorted(degree_centrality.items(), 
                                      key=lambda x: x[1], reverse=True)[:top_k]
        metrics['介数中心性前10'] = sorted(betweenness_centrality.items(), 
                                       key=lambda x: x[1], reverse=True)[:top_k]
        metrics['接近中心性前10'] = sorted(closeness_centrality.items(), 
                                       key=lambda x: x[1], reverse=True)[:top_k]
        metrics['特征向量中心性前10'] = sorted(eigenvector_centrality.items(), 
                                         key=lambda x: x[1], reverse=True)[:top_k]
        
        return metrics
    
    def detect_communities(self, method='louvain'):
        """
        社区检测
        
        Args:
            method: 'louvain' 或 'infomap'
        """
        if self.G is None:
            print("请先构建网络")
            return None
        
        G = self.G
        
        if method == 'louvain':
            # Louvain算法
            partition = community_louvain.best_partition(G, weight='weight')
            
        elif method == 'infomap':
            # Infomap算法
            import infomap
            im = infomap.Infomap("--two-level --directed --silent")
            
            # 添加节点
            for node in G.nodes():
                im.add_node(node)
            
            # 添加边
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1.0)
                im.add_link(u, v, weight)
            
            im.run()
            
            # 提取社区
            partition = {}
            for node in im.tree:
                if node.is_leaf:
                    partition[node.node_id] = node.module_id
        
        # 统计社区信息
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        # 计算模块度
        if method == 'louvain':
            modularity = community_louvain.modularity(partition, G, weight='weight')
        else:
            # Infomap的"code length"可以视为模块度的替代
            modularity = im.codelength
        
        self.communities = communities
        self.partition = partition
        
        print(f"社区检测完成 ({method}):")
        print(f"  发现社区数: {len(communities)}")
        print(f"  模块度: {modularity:.4f}")
        
        # 社区规模分布
        comm_sizes = [len(nodes) for nodes in communities.values()]
        print(f"  社区平均大小: {np.mean(comm_sizes):.1f}")
        print(f"  最大社区: {max(comm_sizes)} 个知识点")
        print(f"  最小社区: {min(comm_sizes)} 个知识点")
        
        return communities, modularity
    
    def analyze_hierarchy(self):
        """
        分析网络的层次结构
        核心问题：是否存在"基础-高级"的知识层次？
        """
        if self.G is None:
            print("请先构建网络")
            return None
        
        G = self.G
        
        # 1. 核心-边缘结构分析
        # 使用k-shell分解
        kshell = nx.core_number(G)
        
        # 2. 层次聚类
        # 构建距离矩阵（基于Jaccard距离）
        nodes = list(G.nodes())
        n = len(nodes)
        jaccard_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    jaccard_matrix[i, j] = 0
                else:
                    # 计算Jaccard相似度
                    neighbors_i = set(G.neighbors(nodes[i]))
                    neighbors_j = set(G.neighbors(nodes[j]))
                    
                    if len(neighbors_i | neighbors_j) > 0:
                        jaccard = 1 - len(neighbors_i & neighbors_j) / len(neighbors_i | neighbors_j)
                    else:
                        jaccard = 1
                    jaccard_matrix[i, j] = jaccard
        
        # 层次聚类
        linkage_matrix = linkage(squareform(jaccard_matrix), method='ward')
        
        # 3. 识别层次
        # 通过轮廓系数确定最佳层次数
        from sklearn.metrics import silhouette_score
        
        silhouette_scores = []
        max_clusters = min(20, len(nodes) - 1)
        
        for k in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            if len(set(clusters)) > 1:
                score = silhouette_score(jaccard_matrix, clusters, metric='precomputed')
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        optimal_k = np.argmax(silhouette_scores) + 2 if silhouette_scores else 2
        
        hierarchy_results = {
            'k_shell_values': kshell,
            'linkage_matrix': linkage_matrix,
            'optimal_clusters': optimal_k,
            'silhouette_scores': silhouette_scores,
            'nodes': nodes
        }
        
        self.hierarchy = hierarchy_results
        
        return hierarchy_results
    
    def visualize_network(self, layout='spring', top_n_central=20, figsize=(15, 12)):
        """
        可视化知识点网络
        
        Args:
            layout: 'spring', 'circular', 'kamada_kawai'
            top_n_central: 只显示中心性最高的n个节点
        """
        if self.G is None:
            print("请先构建网络")
            return None
        
        G = self.G
        
        # 如果节点太多，只显示最重要的节点
        if G.number_of_nodes() > 50:
            print(f"节点过多 ({G.number_of_nodes()})，显示中心性最高的 {top_n_central} 个节点")
            degree_centrality = nx.degree_centrality(G)
            top_nodes = sorted(degree_centrality.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n_central]
            top_node_names = [n for n, _ in top_nodes]
            H = G.subgraph(top_node_names)
        else:
            H = G
        
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(H, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(H)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(H)
        else:
            pos = nx.spring_layout(H, seed=42)
        
        # 节点大小（按度）
        node_sizes = [H.degree(node) * 100 + 50 for node in H.nodes()]
        
        # 节点颜色（按社区）
        if hasattr(self, 'partition'):
            colors = [self.partition.get(node, 0) for node in H.nodes()]
            cmap = plt.cm.tab20
        else:
            colors = 'skyblue'
            cmap = None
        
        # 绘制网络
        nx.draw_networkx_nodes(H, pos, node_size=node_sizes, 
                              node_color=colors, cmap=cmap, alpha=0.8)
        
        # 绘制边
        edges = nx.draw_networkx_edges(H, pos, alpha=0.3, width=1)
        
        # 绘制标签
        labels = {node: node for node in H.nodes()}
        nx.draw_networkx_labels(H, pos, labels, font_size=9, font_family='sans-serif')
        
        plt.title('知识点共现网络', fontsize=16, pad=20)
        plt.axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        
        if cmap and hasattr(self, 'communities'):
            # 创建社区图例
            legend_elements = []
            for comm_id in sorted(set(colors)):
                comm_nodes = [node for node, c in zip(H.nodes(), colors) if c == comm_id]
                if comm_nodes:
                    legend_elements.append(
                        Patch(facecolor=cmap(comm_id/len(set(colors))), 
                             label=f'社区{comm_id}: {len(comm_nodes)}个知识点')
                    )
            
            plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('knowledge_network.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def visualize_hierarchical_structure(self, figsize=(20, 8)):
        """可视化层次结构"""
        print(1)
        if self.hierarchy is None:
            print("请先分析层次结构")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. 层次聚类树状图
        ax1 = axes[0]
        dendrogram(self.hierarchy['linkage_matrix'], 
                  labels=self.hierarchy['nodes'],
                  leaf_rotation=90, ax=ax1)
        ax1.set_title('知识点层次聚类树状图', fontsize=14)
        ax1.set_xlabel('知识点')
        ax1.set_ylabel('距离')
        
        # 2. k-shell层次
        ax2 = axes[1]
        kshell_series = pd.Series(self.hierarchy['k_shell_values'])
        kshell_series.sort_values().plot(kind='barh', ax=ax2, color='salmon')
        ax2.set_title('k-shell核心边缘结构', fontsize=14)
        ax2.set_xlabel('k-shell值')
        ax2.set_ylabel('知识点')
        
        # 3. 轮廓系数
        ax3 = axes[2]
        k_values = range(2, len(self.hierarchy['silhouette_scores']) + 2)
        ax3.plot(k_values, self.hierarchy['silhouette_scores'], 'bo-')
        ax3.axvline(self.hierarchy['optimal_clusters'], color='red', 
                   linestyle='--', label=f'最优k={self.hierarchy["optimal_clusters"]}')
        ax3.set_title('不同聚类数的轮廓系数', fontsize=14)
        ax3.set_xlabel('聚类数')
        ax3.set_ylabel('轮廓系数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hierarchy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    # def compare_with_cognitive_theories(self):
    #     """
    #     与认知理论模型比较
        
    #     比较的理论可能包括：
    #     1. 知识空间理论 (Knowledge Space Theory, KST)
    #     2. ACT-R理论的产生式系统
    #     3. 认知诊断模型 (CDM) 的Q矩阵
    #     """
        
    #     if self.G is None or self.communities is None:
    #         print("请先构建网络和检测社区")
    #         return None
        
    #     comparison_results = {
    #         'theory_comparisons': {},
    #         'consistency_scores': {}
    #     }
        
    #     # 1. 与知识空间理论比较
    #     # KST假设：知识结构是部分有序的，存在先决关系
    #     # 我们可以检查网络的传递性
        
    #     G = self.G
        
    #     # 计算传递性（聚类系数的一种扩展）
    #     transitivity_global = nx.transitivity(G)
    #     transitivity_local = nx.average_clustering(G)
        
    #     comparison_results['knowledge_space_theory'] = {
    #         '全局传递性': transitivity_global,
    #         '局部传递性': transitivity_local,
    #         '解释': '高传递性表明知识结构接近偏序关系，符合KST假设' 
    #                 if transitivity_global > 0.3 else
    #                '中等传递性表明部分有序结构' 
    #                 if transitivity_global > 0.1 else
    #                '低传递性表明知识关系较松散'
    #     }
        
    #     # 2. 与认知负荷理论比较
    #     # 检查核心节点是否为基础概念
    #     degree_centrality = nx.degree_centrality(G)
    #     top_core_nodes = sorted(degree_centrality.items(), 
    #                            key=lambda x: x[1], reverse=True)[:5]
        
    #     # 定义基础概念（需要根据实际情况调整）
    #     basic_concepts = {'加法', '减法', '乘法', '除法', '等式', '图形'}
    #     core_basic_ratio = sum(1 for node, _ in top_core_nodes 
    #                           if any(bc in node for bc in basic_concepts)) / len(top_core_nodes)
        
    #     comparison_results['cognitive_load_theory'] = {
    #         '核心节点': [node for node, _ in top_core_nodes],
    #         '基础概念占比': core_basic_ratio,
    #         '解释': '核心节点多为基础概念，符合认知负荷理论的基础性假设' 
    #                 if core_basic_ratio > 0.5 else
    #                '核心节点包含高级概念，表明网络复杂度较高'
    #     }
        
    #     # 4. 与学习路径理论比较
    #     # 检查是否存在明显的层次结构
    #     if self.hierarchy:
    #         hierarchy_strength = np.mean(self.hierarchy['silhouette_scores']) \
    #                             if self.hierarchy['silhouette_scores'] else 0
            
    #         comparison_results['learning_path_theory'] = {
    #             '最优聚类数': self.hierarchy['optimal_clusters'],
    #             '层次结构强度': hierarchy_strength,
    #             '解释': '强层次结构表明明确的学习路径' 
    #                     if hierarchy_strength > 0.5 else
    #                    '中等层次结构表明部分有序路径' 
    #                     if hierarchy_strength > 0.3 else
    #                    '弱层次结构表明灵活的学习路径'
    #         }
        
    #     return comparison_results
    
    # def generate_theoretical_insights(self):
    #     """生成理论洞见"""
        
    #     if self.G is None:
    #         return "网络未构建"
        
    #     # 获取比较结果
    #     comparisons = self.compare_with_cognitive_theories()
        
    #     insights = []
        
    #     # 知识结构类型判断
    #     density = nx.density(self.G)
    #     avg_clustering = nx.average_clustering(self.G)
        
    #     if density > 0.3 and avg_clustering > 0.6:
    #         structure_type = "密集且高度模块化的知识结构"
    #         theory_implication = "支持知识空间理论中的'知识状态'概念，知识元素紧密关联形成稳定结构"
    #     elif density < 0.1 and avg_clustering > 0.4:
    #         structure_type = "稀疏但局部紧密的小世界结构"
    #         theory_implication = "符合ACT-R理论的'产生式系统'，知识通过关键概念连接"
    #     else:
    #         structure_type = "中等密度的知识网络"
    #         theory_implication = "兼具模块化和连通性，支持灵活的知识迁移"
        
    #     insights.append(f"1. 知识结构类型: {structure_type}")
    #     insights.append(f"   理论意义: {theory_implication}")
        
    #     # 社区结构分析
    #     if self.communities:
    #         n_communities = len(self.communities)
    #         avg_size = np.mean([len(c) for c in self.communities.values()])
            
    #         insights.append(f"\n2. 社区结构: 发现{n_communities}个知识模块")
    #         insights.append(f"   平均大小: {avg_size:.1f}个知识点")
            
    #         if comparisons and 'subject_structure' in comparisons:
    #             purity = comparisons['subject_structure']['平均类别纯度']
    #             if purity > 0.7:
    #                 insights.append("   理论意义: 模块与学科分类高度一致，支持领域特异性认知理论")
    #             elif purity > 0.4:
    #                 insights.append("   理论意义: 中等一致性，表明跨领域知识整合")
    #             else:
    #                 insights.append("   理论意义: 低一致性，表明知识点重组形成新的认知单元")
        
    #     # 层次结构分析
    #     if self.hierarchy and 'learning_path_theory' in comparisons:
    #         strength = comparisons['learning_path_theory']['层次结构强度']
    #         insights.append(f"\n3. 层次结构强度: {strength:.3f}")
            
    #         if strength > 0.5:
    #             insights.append("   理论意义: 强层次性，支持'学习进阶'理论，存在明确的基础-高级序列")
    #         elif strength > 0.3:
    #             insights.append("   理论意义: 中等层次性，部分有序，支持'最近发展区'概念")
    #         else:
    #             insights.append("   理论意义: 弱层次性，支持'建构主义'的灵活知识构建")
        
    #     # 核心节点分析
    #     if comparisons and 'cognitive_load_theory' in comparisons:
    #         basic_ratio = comparisons['cognitive_load_theory']['基础概念占比']
    #         core_nodes = comparisons['cognitive_load_theory']['核心节点'][:3]
            
    #         insights.append(f"\n4. 核心节点: {', '.join(core_nodes)}")
    #         insights.append(f"   基础概念占比: {basic_ratio:.1%}")
            
    #         if basic_ratio > 0.6:
    #             insights.append("   理论意义: 核心多为基础概念，符合'认知负荷理论'的基础支撑作用")
    #         else:
    #             insights.append("   理论意义: 核心包含高级概念，表明复杂知识的中心地位")
        
    #     return "\n".join(insights)
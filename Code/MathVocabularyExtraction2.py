import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Any
import re
import jieba
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class BERTVocabularyDiscoverer:
    """基于BERT的数学词汇发现器"""
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.mlm_model = BertForMaskedLM.from_pretrained(model_name)
            
            self.model.eval()
            self.mlm_model.eval()
            
            # 基础种子词汇
            self.unit_seeds = {'个', '只', '条', '张', '本', '块', '元', '岁', '米', '千克'}
            self.comparison_seeds = {'多', '少', '增加', '减少', '倍', '比'}
            
            logger.info(f"BERT词汇发现器初始化成功，使用模型: {model_name}")
        except Exception as e:
            logger.error(f"BERT模型加载失败: {e}")
            raise
    
    def discover_units_from_corpus(self, texts: List[str], 
                                 top_k: int = 50,
                                 similarity_threshold: float = 0.7) -> List[Dict]:
        """从语料库中发现新的量词/单位词"""
        logger.info(f"开始从{len(texts)}个文本中发现量词...")
        
        # 步骤1: 提取候选量词
        candidates = self._extract_unit_candidates(texts)
        logger.info(f"提取到{len(candidates)}个候选量词")
        
        if not candidates:
            return []
        
        # 步骤2: 获取BERT嵌入表示
        candidate_embeddings = self._get_embeddings(list(candidates.keys()))
        
        # 步骤3: 基于种子词汇的相似性筛选
        seed_embeddings = self._get_embeddings(list(self.unit_seeds))
        
        # 计算每个候选词与种子词汇的平均相似度
        candidate_scores = {}
        for cand, cand_emb in candidate_embeddings.items():
            similarities = []
            for seed_emb in seed_embeddings.values():
                if cand_emb is not None and seed_emb is not None:
                    sim = cosine_similarity([cand_emb], [seed_emb])[0][0]
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                frequency = candidates[cand]
                # 综合评分：相似度 * log(频率) 避免频率主导
                score = avg_similarity * np.log1p(frequency)
                candidate_scores[cand] = {
                    'score': score,
                    'similarity': avg_similarity,
                    'frequency': frequency
                }
        
        # 步骤4: 筛选和排序
        filtered_candidates = []
        for cand, scores in candidate_scores.items():
            if (scores['similarity'] >= similarity_threshold and 
                len(cand) <= 4 and  # 量词通常较短
                scores['frequency'] >= 2):  # 至少出现2次
                filtered_candidates.append((cand, scores))
        
        # 按评分排序
        filtered_candidates.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # 返回top_k个结果
        results = []
        for cand, scores in filtered_candidates[:top_k]:
            # 验证确实是量词（通过上下文分析）
            if self._validate_unit(cand, texts):
                results.append({
                    'word': cand,
                    'score': float(scores['score']),
                    'similarity': float(scores['similarity']),
                    'frequency': scores['frequency'],
                    'type': 'discovered_unit'
                })
        
        logger.info(f"发现{len(results)}个新的量词")
        return results
    
    def _extract_unit_candidates(self, texts: List[str]) -> Dict[str, int]:
        """提取候选量词"""
        candidates = Counter()
        unit_patterns = [
            r'(\d+)([^一二三四五六七八九十百千万亿\s]{1,4})[^\w]',  # 数字+1-4个非数字字符
            r'每([^一二三四五六七八九十百千万亿\s]{1,3})',  # 每+量词
            r'([一二三四五六七八九十百千万亿]+)([^一二三四五六七八九十百千万亿\s]{1,4})[^\w]'  # 中文数字+量词
        ]
        
        for text in texts:
            # 方法1: 正则表达式匹配
            for pattern in unit_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    if len(match.groups()) >= 2:
                        unit_candidate = match.group(2)
                        if self._is_valid_unit_candidate(unit_candidate):
                            candidates[unit_candidate] += 1
            
            # 方法2: 分词+规则（量词通常在数字后面）
            words = jieba.lcut(text)
            for i, word in enumerate(words):
                if i > 0 and self._looks_like_number(words[i-1]):
                    if len(word) <= 3 and self._is_valid_unit_candidate(word):
                        candidates[word] += 1
        
        return dict(candidates)
    
    def _is_valid_unit_candidate(self, word: str) -> bool:
        """判断是否为有效的量词候选"""
        if not word or len(word) > 4:
            return False
        
        # 排除常见非量词
        invalid_chars = {'的', '和', '与', '或', '在', '有', '是', '了', '着', '过'}
        if word in invalid_chars:
            return False
        
        # 排除标点符号
        if re.match(r'^[，。！？；："「」『』（）【】《》]+$', word):
            return False
        
        return True
    
    def _looks_like_number(self, word: str) -> bool:
        """判断是否像数字"""
        number_patterns = [
            r'^\d+$',  # 阿拉伯数字
            r'^\d+\.\d+$',  # 小数
            r'^[一二三四五六七八九十百千万亿]+$',  # 中文数字
            r'^[半两几数若干多少]$'  # 模糊数量词
        ]
        
        for pattern in number_patterns:
            if re.match(pattern, word):
                return True
        return False
    
    def _get_embeddings(self, words: List[str]) -> Dict[str, np.ndarray]:
        """获取词语的BERT嵌入表示"""
        embeddings = {}
        
        for word in words:
            try:
                # 将词语转换为BERT输入
                inputs = self.tokenizer(word, return_tensors="pt", 
                                      padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS] token的嵌入或平均pooling
                    word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                embeddings[word] = word_embedding
            except Exception as e:
                logger.warning(f"获取词语'{word}'的嵌入失败: {e}")
                embeddings[word] = None
        
        return embeddings
    
    def _validate_unit(self, candidate: str, texts: List[str]) -> bool:
        """验证候选词确实是量词"""
        # 检查在上下文中的使用模式
        patterns_found = 0
        
        for text in texts:
            if candidate in text:
                # 检查是否出现在数字后面
                if re.search(rf'\d+{re.escape(candidate)}', text):
                    patterns_found += 1
                # 检查是否出现在"每"后面
                elif re.search(rf'每{re.escape(candidate)}', text):
                    patterns_found += 1
        
        # 至少在2个不同上下文中符合量词模式
        return patterns_found >= 2
    
    def discover_comparison_words(self, texts: List[str], 
                                top_k: int = 30,
                                similarity_threshold: float = 0.6) -> List[Dict]:
        """发现新的比较关系词"""
        logger.info(f"开始从{len(texts)}个文本中发现比较词...")
        
        # 步骤1: 提取候选比较词
        candidates = self._extract_comparison_candidates(texts)
        logger.info(f"提取到{len(candidates)}个候选比较词")
        
        if not candidates:
            return []
        
        # 步骤2: 获取BERT嵌入
        candidate_embeddings = self._get_embeddings(list(candidates.keys()))
        seed_embeddings = self._get_embeddings(list(self.comparison_seeds))
        
        # 步骤3: 计算相似度
        candidate_scores = {}
        for cand, cand_emb in candidate_embeddings.items():
            if cand_emb is None:
                continue
                
            similarities = []
            for seed_emb in seed_embeddings.values():
                if seed_emb is not None:
                    sim = cosine_similarity([cand_emb], [seed_emb])[0][0]
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                frequency = candidates[cand]
                score = avg_similarity * np.log1p(frequency)
                
                candidate_scores[cand] = {
                    'score': score,
                    'similarity': avg_similarity,
                    'frequency': frequency
                }
        
        # 步骤4: 筛选和排序
        filtered_candidates = []
        for cand, scores in candidate_scores.items():
            if (scores['similarity'] >= similarity_threshold and
                len(cand) <= 4 and  # 比较词通常较短
                self._is_valid_comparison_candidate(cand)):
                filtered_candidates.append((cand, scores))
        
        filtered_candidates.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # 步骤5: 分类比较词类型
        results = []
        for cand, scores in filtered_candidates[:top_k]:
            if self._validate_comparison(cand, texts):
                comp_type = self._classify_comparison_type(cand)
                results.append({
                    'word': cand,
                    'score': float(scores['score']),
                    'similarity': float(scores['similarity']),
                    'frequency': scores['frequency'],
                    'type': comp_type
                })
        
        logger.info(f"发现{len(results)}个新的比较词")
        return results
    
    def _extract_comparison_candidates(self, texts: List[str]) -> Dict[str, int]:
        """提取候选比较词"""
        candidates = Counter()
        
        # 比较词常见的上下文模式
        comparison_patterns = [
            r'比.*?([^，。！？]{1,3})\d+',  # 比...X3
            r'([^，。！？]{1,3})出?\d+',     # X出3 / X3
            r'([^，。！？]{1,3})了\d+',     # X了3
            r'是.*的\d+([^，。！？]{1,2})',  # 是...的3X
        ]
        
        for text in texts:
            # 模式匹配
            for pattern in comparison_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    if len(match.groups()) >= 1:
                        candidate = match.group(1)
                        if self._is_valid_comp_candidate(candidate):
                            candidates[candidate] += 1
            
            # 基于"比"字结构的提取
            if '比' in text:
                # 寻找"比"字前后的候选词
                bi_pos = text.find('比')
                # "比"前面的词
                if bi_pos > 0:
                    prev_chars = text[max(0, bi_pos-3):bi_pos]
                    for char in prev_chars:
                        if self._is_valid_comp_candidate(char):
                            candidates[char] += 1
                # "比"后面的词
                if bi_pos < len(text) - 1:
                    next_chars = text[bi_pos+1:min(len(text), bi_pos+4)]
                    for char in next_chars:
                        if self._is_valid_comp_candidate(char):
                            candidates[char] += 1
        
        return dict(candidates)
    
    def _is_valid_comp_candidate(self, word: str) -> bool:
        """判断是否为有效的比较词候选"""
        if not word or len(word) > 4:
            return False
        
        # 排除常见非比较词
        invalid_words = {'的', '了', '在', '有', '是', '和', '与', '或'}
        if word in invalid_words:
            return False
        
        # 排除标点符号和数字
        if re.match(r'^[，。！？；："「」『』（）【】《》\d]+$', word):
            return False
        
        return True
    
    def _is_valid_comparison_candidate(self, candidate: str) -> bool:
        """验证比较词候选的合理性"""
        # 检查是否包含比较语义的常见字符
        comparison_chars = {'多', '少', '增', '减', '加', '高', '低', '长', '短', '大', '小'}
        return any(char in candidate for char in comparison_chars)
    
    def _validate_comparison(self, candidate: str, texts: List[str]) -> bool:
        """验证候选词确实是比较词"""
        patterns_found = 0
        
        for text in texts:
            if candidate in text:
                # 检查是否出现在比较结构中
                if (f'比{candidate}' in text or 
                    f'{candidate}出' in text or 
                    f'{candidate}了' in text or
                    re.search(rf'{re.escape(candidate)}\d+', text)):
                    patterns_found += 1
        
        return patterns_found >= 2
    
    def _classify_comparison_type(self, word: str) -> str:
        """分类比较词类型"""
        increase_indicators = {'多', '增', '加', '高', '长'}
        decrease_indicators = {'少', '减', '低', '短'}
        
        if any(indicator in word for indicator in increase_indicators):
            return 'increase'
        elif any(indicator in word for indicator in decrease_indicators):
            return 'decrease'
        elif '倍' in word or '份' in word:
            return 'ratio'
        else:
            return 'comparison'
    
    def discover_with_mlm(self, texts: List[str], 
                         patterns: List[str],
                         top_k: int = 10) -> List[Dict]:
        """使用掩码语言模型发现新词汇"""
        discovered_words = []
        
        for pattern in patterns:
            try:
                # 将模式中的占位符替换为[MASK]
                masked_text = pattern.replace('[MASK]', self.tokenizer.mask_token)
                
                inputs = self.tokenizer(masked_text, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.mlm_model(**inputs)
                    predictions = outputs.logits
                
                # 获取[MASK]位置的概率分布
                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                mask_logits = predictions[0, mask_token_index, :]
                
                # 获取top_k个预测结果
                top_k_tokens = torch.topk(mask_logits, top_k, dim=1).indices[0].tolist()
                
                for token_id in top_k_tokens:
                    token = self.tokenizer.decode([token_id]).strip()
                    if token and len(token) <= 3:  # 只保留较短的词
                        # 计算概率得分
                        probability = torch.softmax(mask_logits, dim=-1)[0, token_id].item()
                        
                        discovered_words.append({
                            'word': token,
                            'pattern': pattern,
                            'probability': probability,
                            'method': 'mlm'
                        })
            
            except Exception as e:
                logger.warning(f"MLM模式'{pattern}'处理失败: {e}")
                continue
        
        # 去重并按概率排序
        unique_words = {}
        for word_info in discovered_words:
            word = word_info['word']
            if word not in unique_words or word_info['probability'] > unique_words[word]['probability']:
                unique_words[word] = word_info
        
        return sorted(unique_words.values(), key=lambda x: x['probability'], reverse=True)

# 高级词汇发现系统
class AdvancedMathVocabularyDiscoverer:
    """高级数学词汇发现系统"""
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        self.discoverer = BERTVocabularyDiscoverer(model_name)
        
        # MLM发现模式
        self.unit_patterns = [
            "苹果3[MASK]",
            "价格50[MASK]",
            "每[MASK]5元",
            "高度15[MASK]",
            "重量2[MASK]"
        ]
        
        self.comparison_patterns = [
            "小明比小红[MASK]3个苹果",
            "数量[MASK]了5本",
            "价格[MASK]出10元",
            "速度提高了[MASK]2倍"
        ]
    
    def comprehensive_discovery(self, texts: List[str]) -> Dict:
        """综合词汇发现"""
        logger.info("开始综合词汇发现...")
        
        results = {}
        
        # 1. 基于相似度的单位词发现
        units_similarity = self.discoverer.discover_units_from_corpus(texts)
        
        # 2. 基于相似度的比较词发现
        comparisons_similarity = self.discoverer.discover_comparison_words(texts)
        
        # 3. 基于MLM的发现
        units_mlm = self.discoverer.discover_with_mlm(texts, self.unit_patterns)
        comparisons_mlm = self.discoverer.discover_with_mlm(texts, self.comparison_patterns)
        
        # 4. 合并结果
        results['units'] = self._merge_discovery_results(units_similarity, units_mlm, 'unit')
        results['comparisons'] = self._merge_discovery_results(comparisons_similarity, comparisons_mlm, 'comparison')
        
        # 5. 后处理：聚类相似词汇
        results['unit_clusters'] = self._cluster_similar_words([item['word'] for item in results['units']])
        results['comparison_clusters'] = self._cluster_similar_words([item['word'] for item in results['comparisons']])
        
        logger.info(f"发现完成: {len(results['units'])}个单位词, {len(results['comparisons'])}个比较词")
        return results
    
    def _merge_discovery_results(self, similarity_results: List[Dict], 
                               mlm_results: List[Dict], word_type: str) -> List[Dict]:
        """合并不同方法的发现结果"""
        merged = {}
        
        # 添加相似度方法的结果
        for item in similarity_results:
            word = item['word']
            merged[word] = {
                'word': word,
                'score': item.get('score', item.get('similarity', 0)),
                'frequency': item.get('frequency', 0),
                'methods': ['similarity'],
                'type': word_type
            }
        
        # 添加MLM方法的结果
        for item in mlm_results:
            word = item['word']
            if word in merged:
                # 合并得分（平均）
                merged[word]['score'] = (merged[word]['score'] + item['probability']) / 2
                merged[word]['methods'].append('mlm')
            else:
                merged[word] = {
                    'word': word,
                    'score': item['probability'],
                    'frequency': 0,
                    'methods': ['mlm'],
                    'type': word_type
                }
        
        # 转换为列表并按得分排序
        return sorted(merged.values(), key=lambda x: x['score'], reverse=True)
    
    def _cluster_similar_words(self, words: List[str]) -> List[List[str]]:
        """聚类相似词汇"""
        if not words:
            return []
        
        # 获取词向量
        embeddings = []
        valid_words = []
        
        word_embeddings = self.discoverer._get_embeddings(words)
        for word, emb in word_embeddings.items():
            if emb is not None:
                embeddings.append(emb)
                valid_words.append(word)
        
        if len(embeddings) < 2:
            return [[word] for word in words]
        
        # 使用DBSCAN聚类
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.6, min_samples=1).fit(embeddings_array)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for word, label in zip(valid_words, clustering.labels_):
            clusters[label].append(word)
        
        return list(clusters.values())
    
    def generate_discovery_report(self, results: Dict) -> str:
        """生成发现报告"""
        report = []
        report.append("=" * 60)
        report.append("数学词汇自动发现报告")
        report.append("=" * 60)
        
        # 单位词报告
        units = results.get('units', [])
        report.append(f"\n1. 发现的单位词 (共{len(units)}个):")
        for i, unit in enumerate(units[:20], 1):
            methods = ", ".join(unit.get('methods', []))
            report.append(f"   {i:2d}. {unit['word']:4s} (得分: {unit['score']:.3f}, 方法: {methods})")
        
        # 比较词报告
        comps = results.get('comparisons', [])
        report.append(f"\n2. 发现的比较词 (共{len(comps)}个):")
        for i, comp in enumerate(comps[:20], 1):
            methods = ", ".join(comp.get('methods', []))
            report.append(f"   {i:2d}. {comp['word']:4s} (得分: {comp['score']:.3f}, 方法: {methods}, 类型: {comp.get('type', 'unknown')})")
        
        # 聚类报告
        unit_clusters = results.get('unit_clusters', [])
        report.append(f"\n3. 单位词聚类 (共{len(unit_clusters)}个类别):")
        for i, cluster in enumerate(unit_clusters, 1):
            report.append(f"   类别{i}: {', '.join(cluster)}")
        
        comp_clusters = results.get('comparison_clusters', [])
        report.append(f"\n4. 比较词聚类 (共{len(comp_clusters)}个类别):")
        for i, cluster in enumerate(comp_clusters, 1):
            report.append(f"   类别{i}: {', '.join(cluster)}")
        
        return "\n".join(report)
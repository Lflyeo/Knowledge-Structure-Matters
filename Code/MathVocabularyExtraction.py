import hanlp
from hanlp_common.document import Document
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Any
import logging

logger = logging.getLogger(__name__)

class MathVocabularyExtractor:
    """基于HanLP的数学词汇提取器"""
    
    def __init__(self):
        """初始化HanLP管道"""
        try:
            # 创建多任务管道
            self.pipeline = hanlp.pipeline() \
                .append(hanlp.load('COARSE_ELECTRA_SMALL_ZH'), output_key='tok') \
                .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
                .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
                .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')\
                .append(hanlp.load('CPB3_SRL_ELECTRA_SMALL'), output_key='srl',input_key='tok')
            logger.info("HanLP管道初始化成功")
        except Exception as e:
            logger.error(f"HanLP初始化失败: {e}")
            raise
    
    def extract_units_from_dataset(self, dataset: List[str]) -> Dict[str, Any]:
        """从数据集中提取单位词"""
        logger.info(f"开始从{len(dataset)}个文本中提取单位词...")
        
        all_units = []
        unit_contexts = defaultdict(list)
        
        for i, text in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"处理第{i}个文本...")
            
            try:
                units_in_text = self._extract_units_from_text(text)
                all_units.extend(units_in_text)
                
                # 记录每个单位的上下文
                for unit_info in units_in_text:
                    unit_contexts[unit_info['unit']].append({
                        'text': text,
                        'position': unit_info['position'],
                        'context': unit_info['context']
                    })
            except Exception as e:
                logger.warning(f"文本处理失败: {text[:50]}... 错误: {e}")
                continue
        
        # 统计分析
        unit_counter = Counter([info['unit'] for info in all_units])
        
        # 构建单位词典
        unit_lexicon = self._build_unit_lexicon(unit_counter, min_frequency=2)
        
        results = {
            'all_units': all_units,
            'unit_lexicon': unit_lexicon,
            'statistics': {
                'total_units': len(all_units),
                'unique_units': len(unit_counter),
                'unit_frequency': dict(unit_counter.most_common(50)),
                'unit_contexts': dict(unit_contexts)
            }
        }
        
        logger.info(f"单位词提取完成: 共{len(all_units)}个单位, {len(unit_counter)}个唯一单位")
        return results
    
    def _extract_units_from_text(self, text: str) -> List[Dict]:
        """从单个文本中提取单位词"""
        units = []
        
        try:
            # 使用HanLP分析文本
            doc = self.pipeline(text)
            
            # 方法1: 基于词性标注提取量词 (词性标签 'q' 表示量词)
            if 'pos' in doc and 'tok' in doc:
                units.extend(self._extract_units_by_pos(doc['tok'], doc['pos'], text))
            
            # 方法2: 基于依存关系提取数量结构
            if 'dep' in doc and 'tok' in doc:
                units.extend(self._extract_units_by_dependency(doc['tok'], doc['dep'], text))
            
            # 方法3: 基于语义角色标注提取
            if 'srl' in doc and 'tok' in doc:
                units.extend(self._extract_units_by_srl(doc['tok'], doc['srl'], text))
            
            # 方法4: 正则表达式补充提取
            units.extend(self._extract_units_by_regex(text))
            
        except Exception as e:
            logger.warning(f"文本分析失败: {e}")
            # 使用回退的正则方法
            units.extend(self._extract_units_by_regex(text))
        
        # 去重
        return self._deduplicate_units(units)
    
    def _extract_units_by_pos(self, tokens: List[str], pos_tags: List[str], text: str) -> List[Dict]:
        """基于词性标注提取单位词"""
        units = []
        
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos == 'q':  # 量词
                # 检查是否在数字后面
                if i > 0 and self._is_number_like(tokens[i-1]):
                    unit_info = {
                        'unit': token,
                        'position': text.find(token),
                        'context': self._get_token_context(text, token),
                        'method': 'pos_tag',
                        'pattern': f"数字+{token}"
                    }
                    units.append(unit_info)
        
        return units
    
    def _extract_units_by_dependency(self, tokens: List[str], dep_tree: List, text: str) -> List[Dict]:
        """基于依存关系提取单位词"""
        units = []
        
        # 构建依存关系映射
        dep_map = {}
        for head_idx, rel, dep_idx in dep_tree:
            if head_idx != 0:  # 忽略ROOT
                # 调整为0-based索引
                dep_map[dep_idx - 1] = (head_idx - 1, rel)
        
        # 寻找数量修饰关系 (nummod:quantifier)
        for i, token in enumerate(tokens):
            if i in dep_map:
                head_idx, rel = dep_map[i]
                if rel == 'nummod' and head_idx < len(tokens):
                    # i是数字，head_idx是量词
                    if self._is_number_like(token) and self._is_unit_like(tokens[head_idx]):
                        unit_info = {
                            'unit': tokens[head_idx],
                            'position': text.find(tokens[head_idx]),
                            'context': f"{token}{tokens[head_idx]}",
                            'method': 'dependency',
                            'relation': rel
                        }
                        units.append(unit_info)
        
        return units
    
    def _extract_units_by_srl(self, tokens: List[str], srl_spans: List, text: str) -> List[Dict]:
        """基于语义角色标注提取单位词"""
        units = []
        
        for span in srl_spans:
            predicate = span[0]  # 谓词索引
            arguments = span[1:]  # 论元列表
            
            for arg in arguments:
                role, start, end = arg[0], arg[1], arg[2]
                
                # 检查论元中是否包含单位词
                for i in range(start, end + 1):
                    if i < len(tokens) and self._is_unit_like(tokens[i]):
                        # 检查附近是否有数字
                        context_start = max(0, i-2)
                        context_end = min(len(tokens), i+3)
                        context_tokens = tokens[context_start:context_end]
                        
                        if any(self._is_number_like(tok) for tok in context_tokens):
                            unit_info = {
                                'unit': tokens[i],
                                'position': text.find(tokens[i]),
                                'context': ' '.join(context_tokens),
                                'method': 'srl',
                                'role': role
                            }
                            units.append(unit_info)
        
        return units
    
    def _extract_units_by_regex(self, text: str) -> List[Dict]:
        """基于正则表达式提取单位词"""
        units = []
        
        # 模式1: 数字+单位
        patterns = [
            r'(\d+)([个只条张本块元岁米厘米千克克斤两分钟小时天年月周辆架艘头匹棵朵片颗分度秒页章节层倍])',
            r'([一二三四五六七八九十百千万亿]+)([个只条张本块元岁米厘米千克克斤两分钟小时天年月周辆架艘头匹棵朵片颗分度秒页章节层倍])',
            r'每([个只条张本块元岁米厘米千克克斤两分钟小时天年月周辆架艘头匹棵朵片颗分度秒页章节层倍])',
            r'(\d+)(平方米|立方米|平方公里|千克/米|元/千克)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    unit = match.group(2) if match.group(2) else match.group(1)
                    unit_info = {
                        'unit': unit,
                        'position': match.start(2) if match.group(2) else match.start(1),
                        'context': match.group(),
                        'method': 'regex'
                    }
                    units.append(unit_info)
        
        return units
    
    def extract_comparison_words_from_dataset(self, dataset: List[str]) -> Dict[str, Any]:
        """从数据集中提取比较关系词"""
        logger.info(f"开始从{len(dataset)}个文本中提取比较关系词...")
        
        all_comparisons = []
        comparison_contexts = defaultdict(list)
        
        for i, text in enumerate(dataset):
            if i % 100 == 0:
                logger.info(f"处理第{i}个文本...")
            
            try:
                comparisons_in_text = self._extract_comparisons_from_text(text)
                all_comparisons.extend(comparisons_in_text)
                
                # 记录每个比较词的上下文
                for comp_info in comparisons_in_text:
                    comparison_contexts[comp_info['word']].append({
                        'text': text,
                        'position': comp_info['position'],
                        'context': comp_info['context'],
                        'type': comp_info.get('type', 'unknown')
                    })
            except Exception as e:
                logger.warning(f"文本处理失败: {text[:50]}... 错误: {e}")
                continue
        
        # 统计分析
        comp_counter = Counter([info['word'] for info in all_comparisons])
        type_counter = Counter([info.get('type', 'unknown') for info in all_comparisons])
        
        results = {
            'all_comparisons': all_comparisons,
            'comparison_lexicon': set(comp_counter.keys()),
            'statistics': {
                'total_comparisons': len(all_comparisons),
                'unique_comparisons': len(comp_counter),
                'comparison_frequency': dict(comp_counter.most_common(50)),
                'type_distribution': dict(type_counter),
                'comparison_contexts': dict(comparison_contexts)
            }
        }
        
        logger.info(f"比较词提取完成: 共{len(all_comparisons)}个比较, {len(comp_counter)}个唯一比较词")
        return results
    
    def _extract_comparisons_from_text(self, text: str) -> List[Dict]:
        """从单个文本中提取比较关系词"""
        comparisons = []
        
        try:
            doc = self.pipeline(text)
            
            # 方法1: 基于词性标注提取比较词
            if 'pos' in doc and 'tok' in doc:
                comparisons.extend(self._extract_comparisons_by_pos(doc['tok'], doc['pos'], text))
            
            # 方法2: 基于依存关系提取比较结构
            if 'dep' in doc and 'tok' in doc:
                comparisons.extend(self._extract_comparisons_by_dependency(doc['tok'], doc['dep'], text))
            
            # 方法3: 基于语义角色标注提取比较关系
            if 'srl' in doc and 'tok' in doc:
                comparisons.extend(self._extract_comparisons_by_srl(doc['tok'], doc['srl'], text))
            
            # 方法4: 正则表达式补充提取
            comparisons.extend(self._extract_comparisons_by_regex(text))
            
        except Exception as e:
            logger.warning(f"文本分析失败: {e}")
            comparisons.extend(self._extract_comparisons_by_regex(text))
        
        # 去重
        return self._deduplicate_comparisons(comparisons)
    
    def _extract_comparisons_by_pos(self, tokens: List[str], pos_tags: List[str], text: str) -> List[Dict]:
        """基于词性标注提取比较词"""
        comparisons = []
        
        # 定义比较词的可能词性
        comparison_pos_tags = {'VA', 'VV', 'AD'}  # 形容词、动词、副词
        
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            if pos in comparison_pos_tags and self._is_comparison_word(token):
                comp_info = {
                    'word': token,
                    'position': text.find(token),
                    'context': self._get_token_context(text, token),
                    'method': 'pos_tag',
                    'type': self._classify_comparison_type(token)
                }
                comparisons.append(comp_info)
        
        return comparisons
    
    def _extract_comparisons_by_dependency(self, tokens: List[str], dep_tree: List, text: str) -> List[Dict]:
        """基于依存关系提取比较结构"""
        comparisons = []
        
        # 构建依存关系映射
        dep_map = {}
        for head_idx, rel, dep_idx in dep_tree:
            if head_idx != 0:
                dep_map[dep_idx - 1] = (head_idx - 1, rel)
        
        # 寻找比较关系
        for i, token in enumerate(tokens):
            if token == '比' and i in dep_map:
                head_idx, rel = dep_map[i]
                # "比"通常作为介词，连接两个比较对象
                if rel == 'case' and head_idx < len(tokens):
                    # 寻找比较的结果（多/少等）
                    for j in range(i+1, min(i+4, len(tokens))):
                        if self._is_comparison_result(tokens[j]):
                            comp_structure = f"{tokens[head_idx]}比...{tokens[j]}"
                            comp_info = {
                                'word': tokens[j],
                                'position': text.find(tokens[j]),
                                'context': comp_structure,
                                'method': 'dependency',
                                'structure': '比字句',
                                'type': self._classify_comparison_type(tokens[j])
                            }
                            comparisons.append(comp_info)
                            break
        
        return comparisons
    
    def _extract_comparisons_by_srl(self, tokens: List[str], srl_spans: List, text: str) -> List[Dict]:
        """基于语义角色标注提取比较关系"""
        comparisons = []
        
        for span in srl_spans:
            predicate_idx = span[0]
            arguments = span[1:]
            
            # 检查谓词是否为比较词
            if predicate_idx < len(tokens):
                predicate = tokens[predicate_idx]
                if self._is_comparison_word(predicate):
                    comp_info = {
                        'word': predicate,
                        'position': text.find(predicate),
                        'context': self._get_token_context(text, predicate, window=10),
                        'method': 'srl',
                        'type': self._classify_comparison_type(predicate)
                    }
                    comparisons.append(comp_info)
        
        return comparisons
    
    def _extract_comparisons_by_regex(self, text: str) -> List[Dict]:
        """基于正则表达式提取比较关系词"""
        comparisons = []
        
        # 比较模式
        patterns = [
            (r'比.*?([多少增减加减高低长短大小])(?:出|了)?\d*', 'comparison'),
            (r'([多少增减加减高低长短大小])出?\d+', 'difference'),
            (r'是.*的(\d+)倍', 'ratio'),
            (r'([相多]差)\d+', 'difference'),
            (r'([提高降低增加减少])了?\d*', 'change')
        ]
        
        for pattern, comp_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.groups():
                    word = match.group(1)
                    comp_info = {
                        'word': word,
                        'position': match.start(1),
                        'context': match.group(),
                        'method': 'regex',
                        'type': comp_type
                    }
                    comparisons.append(comp_info)
        
        return comparisons
    
    def _is_number_like(self, token: str) -> bool:
        """判断是否像数字"""
        return bool(re.match(r'^(\d+|[一二三四五六七八九十百千万亿半两几数若干多少])$', token))
    
    def _is_unit_like(self, token: str) -> bool:
        """判断是否像单位词"""
        # 常见单位词
        common_units = {'个', '只', '条', '张', '本', '块', '元', '岁', '米', '厘米', 
                       '千克', '克', '斤', '秒', '分钟', '小时', '天', '年', '月', '周'}
        return token in common_units or len(token) == 1  # 单字很可能是单位
    
    def _is_comparison_word(self, token: str) -> bool:
        """判断是否是比较词"""
        comparison_words = {'多', '少', '增加', '减少', '提高', '降低', '倍', '比', 
                          '大于', '小于', '等于', '一样', '相同', '相差'}
        return token in comparison_words
    
    def _is_comparison_result(self, token: str) -> bool:
        """判断是否是比较结果词（多/少等）"""
        result_words = {'多', '少', '大', '小', '高', '低', '长', '短'}
        return token in result_words
    
    def _classify_comparison_type(self, word: str) -> str:
        """分类比较词类型"""
        if word in ['多', '增加', '提高', '大', '高', '长']:
            return 'increase'
        elif word in ['少', '减少', '降低', '小', '低', '短']:
            return 'decrease'
        elif word in ['倍', '比例']:
            return 'ratio'
        elif word in ['等于', '一样', '相同']:
            return 'equal'
        elif word in ['比', '相差']:
            return 'comparison'
        else:
            return 'other'
    
    def _get_token_context(self, text: str, token: str, window: int = 10) -> str:
        """获取词语的上下文"""
        pos = text.find(token)
        if pos == -1:
            return token
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(token) + window)
        return text[start:end]
    
    def _deduplicate_units(self, units: List[Dict]) -> List[Dict]:
        """去重单位词"""
        seen = set()
        unique_units = []
        
        for unit in units:
            key = (unit['unit'], unit['position'])
            if key not in seen:
                seen.add(key)
                unique_units.append(unit)
        
        return unique_units
    
    def _deduplicate_comparisons(self, comparisons: List[Dict]) -> List[Dict]:
        """去重比较词"""
        seen = set()
        unique_comparisons = []
        
        for comp in comparisons:
            key = (comp['word'], comp['position'])
            if key not in seen:
                seen.add(key)
                unique_comparisons.append(comp)
        
        return unique_comparisons
    
    def _build_unit_lexicon(self, unit_counter: Counter, min_frequency: int = 2) -> Set[str]:
        """构建单位词典"""
        return {unit for unit, count in unit_counter.items() 
                if count >= min_frequency and len(unit) <= 4}
    
    def comprehensive_extraction(self, dataset: List[str]) -> Dict[str, Any]:
        """综合提取单位和比较关系词"""
        logger.info("开始综合词汇提取...")
        
        # 提取单位词
        units_result = self.extract_units_from_dataset(dataset)
        
        # 提取比较关系词
        comparisons_result = self.extract_comparison_words_from_dataset(dataset)
        
        # 综合结果
        results = {
            'units': units_result,
            'comparisons': comparisons_result,
            'summary': {
                'total_texts': len(dataset),
                'total_units_found': units_result['statistics']['total_units'],
                'unique_units': units_result['statistics']['unique_units'],
                'total_comparisons_found': comparisons_result['statistics']['total_comparisons'],
                'unique_comparisons': comparisons_result['statistics']['unique_comparisons']
            }
        }
        
        logger.info("综合词汇提取完成")
        return results
    
    def generate_extraction_report(self, results: Dict) -> str:
        """生成提取报告"""
        report = []
        report.append("=" * 60)
        report.append("数学词汇提取报告 (基于HanLP)")
        report.append("=" * 60)
        
        # 单位词报告
        units_stats = results['units']['statistics']
        report.append(f"\n1. 单位词提取结果:")
        report.append(f"   总计提取: {units_stats['total_units']}个单位词")
        report.append(f"   唯一单位: {units_stats['unique_units']}个")
        report.append(f"   高频单位词 (前20):")
        
        for i, (unit, freq) in enumerate(units_stats['unit_frequency'].most_common(20), 1):
            report.append(f"     {i:2d}. {unit}: {freq}次")
        
        # 比较词报告
        comp_stats = results['comparisons']['statistics']
        report.append(f"\n2. 比较关系词提取结果:")
        report.append(f"   总计提取: {comp_stats['total_comparisons']}个比较关系")
        report.append(f"   唯一比较词: {comp_stats['unique_comparisons']}个")
        report.append(f"   类型分布: {comp_stats['type_distribution']}")
        report.append(f"   高频比较词 (前20):")
        
        for i, (comp, freq) in enumerate(comp_stats['comparison_frequency'].most_common(20), 1):
            report.append(f"     {i:2d}. {comp}: {freq}次")
        
        # 总结
        summary = results['summary']
        report.append(f"\n3. 总结:")
        report.append(f"   处理文本数: {summary['total_texts']}")
        report.append(f"   发现单位词: {summary['total_units_found']} (唯一: {summary['unique_units']})")
        report.append(f"   发现比较词: {summary['total_comparisons_found']} (唯一: {summary['unique_comparisons']})")
        
        return "\n".join(report)

    def test_fun(self,text:str):
        doc = self.pipeline(text)
        print(doc)
        return self._extract_units_by_pos(doc['tok'], doc['pos'],text)
import numpy as np
import pandas as pd
import re
from stanfordcorenlp import StanfordCoreNLP

# 加载模型
stanford_model = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05', lang='zh')


class PropertyExtractorClass():
    def __init__(self, unstr_data):
        self.weapon_list = list(unstr_data['weapon'].to_numpy())
        self.ab_list = list(unstr_data['abstract'].to_numpy())
        self.time_list = []
        self.othername_list = []
        self.chara_list = []
        self.country_list = []
        self.func_list = []

        print('----------------- {} -------------------'.format('预定义属性与触发词'))
        self.pre_define()
        print('----------------- {} -------------------'.format('定义完成'))

        print('----------------- {} -------------------'.format('数据清洗'))
        self.data_cleaning()
        print('----------------- {} -------------------'.format('清洗完成'))

        print('----------------- {} -------------------'.format('抽取时间'))
        self.time_extractor()
        print('----------------- {} -------------------'.format('时间抽取完成'))

        print('----------------- {} -------------------'.format('抽取别名'))
        self.othername_extractor()
        print('----------------- {} -------------------'.format('别名抽取完成'))

        print('----------------- {} -------------------'.format('抽取国家'))
        self.country_extractor()
        print('----------------- {} -------------------'.format('国家抽取完成'))

        print('----------------- {} -------------------'.format('抽取特点'))
        self.chara_extractor()
        print('----------------- {} -------------------'.format('特点抽取完成'))

        print('----------------- {} -------------------'.format('抽取功能'))
        self.func_extractor()
        print('----------------- {} -------------------'.format('功能抽取完成'))

        print('----------------- {} -------------------'.format('数据整合与导出'))
        self.write_to_csv()
        print('----------------- {} -------------------'.format('导出完成'))

    def pre_define(self):
        '''
        预定义属性与触发词
        '''
        self.property_columns = [
            '武器名称',
            '武器别名',
            '时间',
            '国家',
            '武器特点',
            '武器作用',
        ]

        # 标点符号
        self.punc = ['，', '。', '？', '！', '、']

        # 武器特点触发词
        self.chara_trigger = [
            '射程', '精度', '具有', '能够', '采用', '特点', '体积', '重量', '全重', '全长', '速度', '行程', '口径'

        ]

        # 武器作用触发词
        self.func_trigger = [
            '加强了', '用于', '用途', '作用', '可为'
        ]

        # 别名触发词
        self.othername_trigger = [
            '绰号名为', '原名：', '原名为', '原名', '又译为：', '又译为', '又译作', '又译', '绰号译文：', '绰号：', '绰号是', '绰号', '又称作', '又称为',
            '又称：', '又称', '称作'

        ]

        # 国家
        self.country_trigger = ['我国', '中国', '蒙古', '朝鲜', '苏联', '韩国', '日本', '越南', '柬埔寨', '老挝', '泰国', '缅甸', '菲律宾', '文莱',
                                '马来西亚', '新加坡', '印度尼西亚', '东帝汶', '巴基斯坦', '印度', '尼泊尔', '不丹', '孟加拉国', '马尔代夫', '斯里兰卡',
                                '哈萨克斯坦', '乌兹别克斯坦', '吉尔吉斯斯坦', '土库曼斯坦', '塔吉克斯坦', '土耳其', '格鲁吉亚', '亚美尼亚', '阿塞拜疆', '伊朗',
                                '阿富汗', '叙利亚', '伊拉克', '沙特阿拉伯', '黎巴嫩', '以色列', '巴勒斯坦', '约旦', '科威特', '巴林', '卡塔尔',
                                '阿拉伯联合酋长国', '也门', '阿曼', '冰岛', '丹麦', '挪威', '瑞典', '芬兰', '英国', '爱尔兰', '荷兰', '比利时', '卢森堡',
                                '法国', '摩纳哥', '德国', '瑞士', '列支敦士登', '波兰', '捷克', '斯洛伐克', '奥地利', '匈牙利', '爱沙尼亚', '拉脱维亚',
                                '立陶宛', '白俄罗斯', '乌克兰', '摩尔多瓦', '俄罗斯', '葡萄牙', '西班牙', '安道尔', '意大利', '圣马力诺', '梵蒂冈', '马耳他',
                                '斯洛文尼亚', '克罗地亚', '波斯尼亚和黑塞哥维那', '黑山', '塞尔维亚', '阿尔巴尼亚', '北马其顿', '保加利亚', '希腊', '罗马尼亚',
                                '塞浦路斯', '\xa0', '埃及', '利比亚', '突尼斯', '阿尔及利亚', '摩洛哥', '尼日尔', '布基纳法索', '马里', '毛里塔尼亚',
                                '尼日利亚', '贝宁', '多哥', '加纳', '科特迪瓦', '利比里亚', '塞拉利昂', '几内亚', '几内亚比绍', '塞内加尔', '冈比亚', '佛得角',
                                '圣多美和普林西比', '喀麦隆', '赤道几内亚', '加蓬', '刚果共和国', '乍得', '中非', '刚果民主共和国', '吉布提', '索马里', '厄立特里亚',
                                '埃塞俄比亚', '苏丹', '南苏丹', '肯尼亚', '坦桑尼亚', '乌干达', '卢旺达', '布隆迪', '塞舌尔', '安哥拉', '赞比亚', '马拉维',
                                '莫桑比克', '纳米比亚', '博茨瓦纳', '津巴布韦', '南非', '斯威士兰', '莱索托', '马达加斯加', '毛里求斯', '科摩罗', '澳大利亚',
                                '新西兰', '帕劳', '密克罗尼西亚联邦', '马绍尔群岛', '瑙鲁', '基里巴斯', '巴布亚新几内亚', '所罗门群岛', '瓦努阿图', '斐济', '图瓦卢',
                                '萨摩亚', '汤加', '纽埃', '库克群岛', '加拿大', '美国', '墨西哥', '危地马拉', '伯利兹', '萨尔瓦多', '洪都拉斯', '尼加拉瓜',
                                '哥斯达黎加', '巴拿马', '巴哈马', '古巴', '牙买加', '海地', '多米尼加共和国', '圣基茨和尼维斯', '安提瓜和巴布达', '多米尼克',
                                '圣卢西亚', '巴巴多斯', '圣文森特和格林纳丁斯', '格林纳达', '特立尼达和多巴哥', '哥伦比亚', '委内瑞拉', '圭亚那', '苏里南', '厄瓜多尔',
                                '秘鲁', '玻利维亚', '巴西', '智利', '阿根廷', '乌拉圭', '巴拉圭']

    def output(self, li):
        '''
        输出函数
        '''
        for i, j in enumerate(li):
            print(i, j)

    def data_cleaning(self):
        '''
        数据清洗
        '''

        # 去转义符
        pat = re.compile('[\\t|\\n|\\r|\\xa0|\\u3000]')
        for i in range(len(self.ab_list)):
            self.ab_list[i] = re.sub(pat, '', self.ab_list[i])
        # print(self.ab_list)

        # 删除异常元素
        expe = '请用一段简单的话描述该词条'
        print('原始个数：{}'.format(len(self.ab_list)))
        i = 0
        del_num = 0
        while i < len(self.ab_list):
            if expe in self.ab_list[i]:
                # print('异常数据为：{}'.format(self.ab_list[i]))
                self.ab_list.pop(i)
                self.weapon_list.pop(i)
                del_num += 1
                # print('已删除{}条'.format(del_num))
                continue
            i += 1

        print('共删除{}条，还剩{}条'.format(del_num, len(self.ab_list)))

    def time_extractor(self):
        '''
        时间抽取
        '''
        # 抽取时间相关语句
        time_pat = re.compile('[^|。].*?\d{4}年.*?。')
        time_p1 = re.compile('[，。].{0,10}?\d{4}年.*?。|^.{0,15}\d{4}年.*?。|。.{0,10}\d{2}年代.*?。')
        s = 0
        time_list1 = []
        for ab in self.ab_list:
            a = re.findall(time_p1, ab)
            b = re.findall(time_pat, ab)
            # print(s, len(a), a)
            # print(s, len(b), b)
            s += 1

            if a:
                if a[0][0] in ['。', '，']:
                    time_list1.append(a[0][1:])
                else:
                    time_list1.append(a[0])
            elif b:
                if b[0][0] in ['。', '，']:
                    time_list1.append(b[0][1:])
                else:
                    time_list1.append(b[0])
            else:
                time_list1.append('无')

        # self.output(time_list1)

        # 词性标注
        time_list2 = []  # 对每个句子进行词性标注
        V_L = []  # 筛选出我们需要的词组
        for t in time_list1:
            p = stanford_model.pos_tag(t)
            time_list2.append(p)
            V = []
            for k in p:
                if k[1] in ['VV', 'NT']:
                    V.append(k)
                if k[0] in ['，', '。']:
                    V.append(k)
            V_L.append(V)

            # self.output(time_list2)
        # self.output(V_L)

        # 时间相关词抽取
        NTVV = []
        for v1 in V_L:
            nv = []
            flag1 = 0
            flag2 = 0
            NT_num = 0
            for v2 in v1:
                if v2[1] == 'NT':
                    NT_num += 1
                    if NT_num == 4:
                        break
                    nv.append(v2)
                    flag1 = 1

                if v2[1] == 'VV' and flag1 == 1:
                    nv.append(v2)
                    flag2 = 1
                if v2[1] == 'PU' and flag1 == 1 and flag2 == 1:
                    break
            NTVV.append(nv)

            # self.output(NTVV)

        # 拼接
        for nv in NTVV:
            time_str = ''
            if nv:
                for n in nv:
                    time_str += n[0]
            else:
                time_str = '暂无介绍'
            self.time_list.append(time_str)

        # self.output(self.time_list)

    def othername_extractor(self):
        '''
        别名抽取
        '''
        punc4 = ['。', '？', '！', '；', '，', ',', '（', '）', '(', ')']
        othername_pat = re.compile('(?:{})(.*)'.format('|'.join(self.othername_trigger)))
        # print(othername_pat)
        for ab in self.ab_list:
            ab2 = re.split('[{}]'.format(''.join(punc4)), ab)
            oth = []
            for sen in ab2:
                temp = re.findall(othername_pat, sen)
                oth.extend(temp)
            if oth and oth[0] == '':
                oth = []
            if oth:
                self.othername_list.append('/'.join(oth))
            else:
                self.othername_list.append('无')

        # self.output(othername_list)

    def country_extractor(self):
        '''
        国家抽取
        '''
        country_pat = re.compile('({})'.format('|'.join(self.country_trigger)))
        # print(country_pat)

        for ab in self.ab_list:
            c = re.findall(country_pat, ab)
            if c:
                if c[0] == '我国':
                    self.country_list.append('中国')
                else:
                    self.country_list.append(c[0])
            else:
                self.country_list.append('未知')

        # self.output(self.country_list)

    def chara_extractor(self):
        '''
        特点抽取
        '''
        punc2 = ['。', '？', '！', '；', '，']

        chara_pat = re.compile('.*(?:{}).*'.format('|'.join(self.chara_trigger)))
        # print(chara_pat)

        for ab in self.ab_list:
            ab_split = re.split('[{}]'.format(''.join(punc2)), ab)
            le = []
            for ab_l in ab_split:
                l = re.findall(chara_pat, ab_l)
                le.extend(l)
            if le:
                self.chara_list.append('，'.join(le))
            else:
                self.chara_list.append('暂无介绍')
        # self.output(self.chara_list)

    def func_extractor(self):
        '''
        作用抽取
        '''
        punc3 = ['。', '？', '！']

        func_pat = re.compile('.*(?:{}).*'.format('|'.join(self.func_trigger)))
        # print(func_pat)

        for ab in self.ab_list:
            ab2 = re.split('[{}]'.format(''.join(punc3)), ab)
            fu = []
            for sen in ab2:
                temp = re.findall(func_pat, sen)
                fu.extend(temp)
            if fu:
                self.func_list.append('，'.join(fu))
            else:
                self.func_list.append('暂无介绍')

        # self.output(self.func_list)

    def write_to_csv(self):
        '''
        数据整合与输出
        '''
        weapon_property = [self.weapon_list, self.othername_list, self.time_list, self.country_list, self.chara_list,
                           self.func_list]
        weapon_data = dict(zip(self.property_columns, weapon_property))
        weapon_pd = pd.DataFrame(weapon_data)
        weapon_pd.to_csv('PropertyExtractor.csv', encoding='utf-8')


if __name__ == '__main__':
    unstr_file_path = 'unstructured_data.csv'
    unstr_data = pd.read_csv(unstr_file_path)
    unstr_data
    extractor = PropertyExtractorClass(unstr_data)
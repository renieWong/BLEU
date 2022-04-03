import sys
import codecs
import os
import math
import operator
import json
from functools import reduce
from nltk import bleu

def fetch_data(cand, ref):
    """
    将每个reference和candidate存储为一个列表
    :param cand: 候选文件名
    :param ref: 参考文件名
    :return:
    """
    references = []
    # 如果参考文件名的后缀为'.txt'，则将参考文件中每一行添加到references列表中
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    # 如果参考文件名的后缀不为'.txt'，即在一个文件夹下，则先找到文件再添加
    else:
        # 返回的是一个三元组(root,dirs,files)，遍历每一个file
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    # 返回由候选文件中每一行构成的列表
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    # 返回参考列表和候选列表
    return candidate, references


# candidate = [["word peace],['make china great again !']]
# reference [["world war"],['make USA great again']]

def count_ngram(candidate, references, n):
    """
    计算n-gram的P_n
    :param candidate: 候选列表
    :param references: 参考列表
    :param n: n-gram
    :return:
    """
    # 初始化
    clipped_count = 0
    # 统计候选集中n-gram的数量
    count = 0
    # 用来记录reference长度
    r = 0
    # 用来记录 candidates的长度
    c = 0
    # 遍历每一个CANDIDATES
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        # 统计ref中的每个n-gram 的数目
        ref_counts = []
        # 统计 REF 的长度，length
        ref_lengths = []

        # 遍历每一个REFERENCE
        # Build dictionary of ngram counts-构建字典统计在references中n-gram的次数
        for reference in references:
            # 对应的参考集
            ref_sentence = reference[si]
            # ngram统计
            ngram_d = {}
            # 将参考集以空格分割
            words = ref_sentence.strip().split()
            # 记录每一个参考集的单词数
            ref_lengths.append(len(words))
            # 参考集中有多少组n-gram
            limits = len(words) - n + 1      # [1,2,3,4,5,6,7]
            # 遍历每组n-gram
            for i in range(limits):
                # 构造n-gram
                ngram = ' '.join(words[i:i + n]).lower()
                # ref中n-gram计数
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            # 将统计参考集n-gram数目的字典添加到列表中
            ref_counts.append(ngram_d)
        # 遍历 CANDIDATE
        cand_sentence = candidate[si]
        # 统计cand中的每个n-gram 的数目
        cand_dict = {}
        # 将候选集以空格进行分割
        words = cand_sentence.strip().split()
        # 候选集对应n-gram的数量
        limits = len(words) - n + 1
        # 遍历n-gram
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            # cand中n-gram计数
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        # 遍历每一个CANDIDATES，累加Count_clip值
        clipped_count += clip_count(cand_dict, ref_counts)
        # 统计候选集中n-gram的数量
        count += limits
        # 计算参考集的句长
        r += best_length_match(ref_lengths, len(words))
        # 计算候选集的句长
        c += len(words)
    # 得到P_n
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    # 计算BP值
    bp = brevity_penalty(c, r)
    # 返回BP与P_n
    return pr, bp


def clip_count(cand_d, ref_ds):
    """
    Count the clip count for each ngram considering all references
    :param cand_d: 候选集中n-gram字典
    :param ref_ds: 多个参考集中n-gram字典
    :return:返回Count_clip值
    """
    # 基于Count_clip公式计算
    count = 0
    for m in cand_d.keys():
        # 候选集中某一个n-gram的次数
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count


def best_length_match(ref_l, cand_l):
    """
    Find the closest length of reference to that of candidate
    :param ref_l: 多个参考集的单词数
    :param cand_l: 候选集的单词数
    :return:返回参考集的句长
    """
    # 初始一个差值
    least_diff = abs(cand_l - ref_l[0])
    best = ref_l[0]
    # 遍历每一个参考集的单词数
    for ref in ref_l:
        # 如果比least_diff小，那么重新赋值
        if abs(cand_l - ref) < least_diff:
            least_diff = abs(cand_l - ref)
            best = ref
    return best


def brevity_penalty(c, r):
    """
    对长度进行惩罚
    :param c: 候选集的句长
    :param r: 参考集的句长
    :return:
    """
    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - (float(r) / c))

    return bp



def geometric_mean(precisions):
    """
    基于BP与P_n计算bleu
    :param precisions: 精确率
    :return: 返回bleu
    """
    # reduce函数:用传给 reduce 中的函数 function（有两个参数）
    # 先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
    # operator.mul
    # exp(\sum W_n log(P_n))
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    """
    计算bleu
    :param candidate:候选列表
    :param references:参考列表
    :return:返回bleu
    """
    precisions = []
    # 从1-gram遍历到4-gram
    for i in range(4):
        # 得到P_n,BP
        pr, bp = count_ngram(candidate, references, i + 1)
        precisions.append(pr)
        print('P' + str(i + 1), ' = ', round(pr, 2))
    print('BP = ', round(bp, 2))
    # 基于BP与P_n计算bleu
    bleu = geometric_mean(precisions) * bp
    return bleu


if __name__ == "__main__":
    # 获取data，返回参考列表和候选列表
    # sys.argv是获取运行python文件的时候命令行参数
    candidate, references = fetch_data(sys.argv[1], sys.argv[2])
    # 计算评价指标bleu
    bleu = BLEU(candidate, references)
    # 将计算得到的bleu写入'bleu_out.txt'文件中
    print('BLEU = ', round(bleu, 4))
    out = open('bleu_out.txt', 'w')
    out.write(str(bleu))
    out.close()

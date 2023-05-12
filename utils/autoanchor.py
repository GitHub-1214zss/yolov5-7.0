# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    '''
    检查YOLOv5 Detect()模块，确定anchors和stride的顺序是一致的，并在必要时进行纠正
    :params m: model中的最后一层 Detect层
    '''
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """
    通过bpr(best possible recall)<98%? 确定是否需要改变anchors 需要就调用k-means重新计算anchors; 
    用于train.py中
    :params dataset: LoadImagesAndLabels返回的数据集
    :params model: 初始化的模型
    :params thr: 超参中得到  界定anchor与label匹配程度的阈值
    :params imgsz: 图片尺寸 默认640
    """
    # m: 从model中取出最后一层(Detect)
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # dataset.shapes.max(1, keepdims=True) = 每张图片的较长边
    # shapes: 将数据集图片的最长边缩放到img_size, 较小边相应缩放 得到新的所有数据集图片的宽高
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 产生随机数scale
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    # 所有target(n张图片的标签)的wh   基于原图大小    shapes * scale: 随机化尺度变化
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """
        用在check_anchors函数中
        根据数据集的所有图片的wh和当前所有anchors k计算 bpr(best possible recall) 和 aat(anchors above threshold)
        :params k: anchors  wh
        :return bpr: best possible recall 最多能被召回(通过thr=4.0)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        :return aat: anchors above threshold 每个target平均有多少个anchors(一般达不到9个，会筛选一部分)
        """
        # None添加维度  所有target(gt)的wh
        #             所有anchor的wh k
        # r: target的高h宽w与anchor的高h_a宽w_a的比值，即h/h_a, w/w_a    有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        # x 高宽比和宽高比的最小值 (无论r大于1，还是小于等于1最后统一结果都要小于1)
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # best为每个gt框选择匹配所有anchors宽高比例值最好的那一个比值
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold 每个target平均有多少个anchors
        # bpr(best possible recall) = 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # 模型步长
    # anchors:  所有anchors的宽高   基于缩放后的图片大小(较长边为640 较小边相应缩放)
    anchors = m.anchors.clone() * stride  # 当前anchor
    # 计算出数据集所有图片的wh和当前所有anchors的bpr和aat
    # bpr:最多能被召回(通过thr)的gt框数量 / 所有gt框数量  小于0.98 才会用k-means计算anchor
    # aat(anchors past thr):通过阈值的anchor个数
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    # 考虑这9类anchor的宽高和gt框的宽高之间的差距, 如果bpr<0.98(说明当前anchor不能很好的匹配数据集gt框)就会根据k-means算法重新聚类新的anchor
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        # 如果bpr<0.98 使用k-means + 遗传进化算法选择出与数据集更匹配的anchors框
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        # 计算新的anchors的new_bpr
        new_bpr = metric(anchors)[0]
        # 比较k-means + 遗传进化算法进化后的anchors的new_bpr和原始anchors的bpr
        # 注意: 这里并不一定进化后的bpr必大于原始anchors的bpr, 因为两者的衡量标注是不一样的  进化算法的衡量标准是适应度 而这里比的是bpr
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            # 替换m的anchor_grid
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            # 检查anchor顺序和stride顺序是否一致 不一致就调整
            # 因为我们的m.anchors是相对各个feature map 所以必须要顺序一致 否则效果会很不好
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """在check_anchors中调用
    使用K-means + 遗传算法 算出更符合当前数据集的anchors

    Arguments:
        path: 数据集路径或者数据集本身
        n: anchor框的个数
        img_size: 数据集图片约定的大小
        thr: 阈值 由hyp['anchor_t']参数指定，默认4.0
        gen: 遗传算法进化迭代的次数(突变 + 选择)
        verbose: 是否打印所有的进化(成功的)结果 默认传入是Fasle的 只打印最佳的进化结果即可
    Return:
        k: k-means + 遗传算法进化 后的anchors
    Usage:
        from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random  # 随机工具
    # 注意一下下面的thr不是传入的thr，而是1/thr, 所以在计算指标这方面还是和check_anchor一样
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        """用于print_results函数和anchor_fitness函数

        计算ratio metric: 整个数据集的gt框与anchor对应宽比和高比即:gt_w/k_w,gt_h/k_h + x + best_x  用于后续计算bpr+aat;

        注意我们这里选择的metric是gt框与anchor对应宽比和高比 而不是常用的iou 这点也与nms的筛选条件对应 是yolov5中使用的新方法.

        :params k: anchor框

        :params wh: 整个数据集的wh 

        :return x:  N个gt框与所有anchor框的宽比或高比(两者之中较小者)

        :return x.max(1)[0]:N个gt框与所有anchor框中的最大宽比或高比(两者之中较小者)
        """
        # 两者的重合程度越高 就越趋近于1 远离1(<1 或 >1)重合程度都越低
        # r=gt_height/anchor_height  gt_width / anchor_width  有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        # torch.min(r, 1. / r):将所有的宽比和高比统一到<=1
        # .min(2): 选出每个gt个和anchor的宽比和高比最小的值
        # [0] 返回 每个gt个和anchor的宽比和高比最小的值 就是所有gt与anchor重合程度最低的
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        # x.max(1)[0] 返回每个gt和所有anchor(9个)中宽比/高比最大的值
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        """用于kmean_anchors函数
        适应度计算 优胜劣汰 用于遗传算法中衡量突变是否有效的标注 如果有效就进行选择操作 没效就继续下一轮的突变
        :params k: k-means生成的9个anchors     wh: : 数据集的所有gt框的宽高
        :return (best * (best > thr).float()).mean()=适应度计算公式  注意和bpr有区别 这里是自定义的一种适应度公式
                返回的是输入此时anchor k 对应的适应度
        """
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        """用于kmean_anchors函数中打印k-means计算相关信息
        计算bpr、aat=>打印信息: 阈值+bpr+aat  anchor个数+图片大小+metric_all+best_mean+past_mean+Kmeans聚类出来的anchor框(四舍五入)
        :params k: k-means得到的anchor k
        :return k: input
        """
        # 将k-means得到的anchor k按面积从小到大啊排序
        k = k[np.argsort(k.prod(1))]  # sort small to large
        # x: N个gt框与所有anchor框的宽比或高比(两者之中较小者)
        # best:N个gt框与所有anchor框中的最大 宽比或高比(两者之中较小者)
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    # 载入数据集
    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        # 加载数据集
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    # 得到数据集中所有数据的wh
    # 将数据集图片的最长边缩放到img_size, 较小边相应缩放
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 将原本数据集中gt boxes归一化的wh缩放到shapes尺度，与check_anchors-wh对比未做随机尺度变化
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    # 统计gt boxes中宽或者高小于3个像素的个数, 目标太小 发出警告
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    # 筛选出label大于2个像素的框拿来聚类,[...]内的相当于一个筛选器,为True的留下
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    # Kmeans聚类方法: 使用欧式距离来进行聚类
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        # 计算宽和高的标准差->[w_std,h_std]
        s = wh.std(0)  # sigmas for whitening
        # 开始聚类,仍然是聚成n类,返回聚类后的anchors k(这个anchor k是白化后数据的anchor框)
        # 另外还要注意的是这里的kmeans使用欧式距离来计算的
        # 运行k-means的次数为30次  obs: 传入的数据必须先白化处理 'whiten operation'
        # 白化处理: 新数据的标准差=1 降低数据之间的相关度，不同数据所蕴含的信息之间的重复性就会降低，网络的训练效率就会提高
        # 白化操作博客: https://blog.csdn.net/weixin_37872766/article/details/102957235
        # k*s 得到原来数据(白化前)的anchor框
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    # 过滤前后，wh过滤了的，wh0未过滤的
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    # 输出新算的anchors k 相关的信息
    k = print_results(k, verbose=False)

    # Evolve
    # Evolve 类似遗传/进化算法  选择+变异操作
    # mp: 突变比例mutation prob=0.9   s: sigma=0.1
    # anchor_fitness(k):适应度计算 优胜劣汰 用于遗传算法中衡量突变是否有效的标注 返回的是输入此时anchor k 对应的适应度
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # 进度条
    # 根据聚类出来的n个点采用遗传算法生成新的anchor
    for _ in pbar:
        # 重复1000次突变+选择 选择出1000次突变里的最佳anchor k和最佳适应度f
        v = np.ones(sh)
        while (v == 1).all():
            # 产生变异规则 修改直到发生变化(防止重复)
            # npr.random(sh) < mp: 让v以90%的比例进行变异
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        # 变异(改变这一时刻之前的最佳适应度对应的anchor k)
        kg = (k.copy() * v).clip(min=2.0)
        # 计算变异后的anchor kg的适应度
        fg = anchor_fitness(kg)
        # 如果变异后的anchor kg的适应度>最佳适应度k 就进行选择操作
        if fg > f:
            # 选择变异后的anchor kg为最佳的anchor k 变异后的适应度fg为最佳适应度f
            f, k = fg, kg.copy()
            # 打印信息
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                # 打印最佳进化结果
                print_results(k, verbose)

    return print_results(k).astype(np.float32)

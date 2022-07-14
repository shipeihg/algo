# -*- coding:utf-8 -*-

'''
堆的概念及和二叉树、数组关系
https://www.jb51.net/article/222484.htm
https://blog.csdn.net/wcc8848/article/details/122283705
堆练习
https://leetcode-cn.com/problems/top-k-frequent-elements/solution/python-dui-pai-xu-by-xxinjiee/
'''




# 215. 数组中的第K个最大元素
# 维护一个size为k小根堆，迭代完了后，这k个数就是数组里面最大的k个数，而堆顶是k个中最小的，即是第k个最大元素
# https://leetcode-cn.com/problems/top-k-frequent-elements/solution/pythonti-jie-xiao-gen-dui-by-xiao-xue-66/
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        # 用于初始化最小堆的函数，注意从中间开始；这个函数的目的是把最小值放到堆顶
        def heapSort(arr):
            for i in range(len(arr) >> 1, -1, -1):  # 从中间节点遍历，保证可以遍历到二叉树的全部，因为父子节点在数组中的索引是2倍关系；倒叙遍历，是为了保证堆顶是最小的元素
                minHeapSink(arr, i, len(arr) - 1)

        # 注意一下，这里的堆就是用数组表示的
        # 递归调整最小堆，让父节点始终小于子节点的最小值，否则就递归
        def minHeapSink(nums, start, end): # 这个是个置换函数，被heapSort调用，保证最小值在start位置
            if 2*start + 1 > end:
                return
            minChild = 2*start+2 if 2*start+2<=end and nums[2*start+2]<nums[2*start+1] else 2*start+1 # 比较左右节点，哪个小取哪个
            if nums[start] > nums[minChild]:
                nums[start], nums[minChild] = nums[minChild], nums[start]
                minHeapSink(nums, minChild, end) # 递归，这里只对以minChild为父节点链路上的节点进行矫正，不是整个树

        ans = nums[:k] # 首先确定最小堆的尺寸
        heapSort(ans) # 建立最小堆
        for num in nums[k:]:
            if num > ans[0]: # 如果一旦当前数字大于堆顶元素，则删除堆顶，吧当前加入堆里调整
                ans[0] = num
                minHeapSink(ans, 0, k-1)
        return ans[0] # 返回堆顶元素



# 295. 数据流的中位数
# https://leetcode.cn/problems/find-median-from-data-stream/solution/tu-jie-pai-xu-er-fen-cha-zhao-you-xian-dui-lie-by-/
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # 初始化大顶堆和小顶堆
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        if len(self.max_heap) == len(self.min_heap):# 先加到大顶堆，再把大堆顶元素加到小顶堆
            heapq.heappush(self.min_heap, -heapq.heappushpop(self.max_heap, -num))
        else:  # 先加到小顶堆，再把小堆顶元素加到大顶堆
            heapq.heappush(self.max_heap, -heapq.heappushpop(self.min_heap, num))

    def findMedian(self) -> float:
        if len(self.min_heap) == len(self.max_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return self.min_heap[0]

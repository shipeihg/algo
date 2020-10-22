# -*- coding:utf-8 -*-

'''
堆练习
https://leetcode-cn.com/problems/top-k-frequent-elements/solution/python-dui-pai-xu-by-xxinjiee/
'''




# 215. 数组中的第K个最大元素
# 维护一个小根堆
# https://leetcode-cn.com/problems/top-k-frequent-elements/solution/pythonti-jie-xiao-gen-dui-by-xiao-xue-66/
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
    # 注意一下，这里的堆就是用数组表示的
    # 递归调整最小堆，让父节点始终小于子节点的最小值，否则就递归 
    def minHeapSink(nums, start, end):
        if 2*start + 1 <= end:
        minChild = 2*start+2 if 2*start+2<=end and nums[2*start+2]<nums[2*start+1] else 2*start+1
        if nums[start] > nums[minChild]:
            nums[start], nums[minChild] = nums[minChild], nums[start]
            minHeapSink(nums, minChild, end) # 递归

    # 用于初始化最小堆的函数，注意从中间开始
    def heapSort(arr):
        for i in range(len(arr)>>1, -1, -1):
            minHeapSink(arr, i, len(arr)-1)

    ans = nums[:k] # 首先确定最小堆的尺寸
    heapSort(ans) # 建立最小堆
    for num in nums[k:]:
        if num > ans[0]: # 如果一旦当前数字大于堆顶元素，则删除堆顶，吧当前加入堆里调整
            ans[0] = num
            minHeapSink(ans, 0, k-1)
    return ans[0] # 返回堆顶元素

s = Solution()
print s.findKthLargest([1,10,2,5,4,7,0], 6)

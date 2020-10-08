# 二分总结模板
# https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/



# 34. 在排序数组中查找元素的第一个和最后一个位置
# https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/si-lu-hen-jian-dan-xi-jie-fei-mo-gui-de-er-fen-cha/
class Solution:
    def searchLeft(self, a, k):
        i, j = 0, len(a)-1
        while i < j:
            m = (i + j) >> 1
            if a[m] < k: i = m + 1
            else: j = m
        return i if a[i]==k else -1
    
    def searchRight(self, a, k):
        i, j = 0, len(a)-1
        while i < j:
            m = (i + j + 1) >> 1
            if a[m] > k: j = m - 1
            else: i = m
        return i 
    
    def searchRange(self, a, k):
        if not a: return [-1,-1]
        left, right = self.searchLeft(a, k), self.searchRight(a, k)
        if left == -1: return [-1,-1]
        return [left, right]


# 寻找旋转排序数组中的最小值 II
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i,j=0,len(nums)-1
        while i<j:
            m=(i+j)>>1
            if nums[m]>nums[j]: i=m+1
            elif nums[m]<nums[j]:j=m # 吧中点作为有边界
            else: j-=1 # 可能存在重复值，nums[m]==nums[j]的时候，指针应该向让nums[j]减小的方向移动
        return nums[i]


# 287. 寻找重复数
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        i,j = 0, len(nums)-1
        while i<j:
            m = (i+j) >> 1
            cnt = sum(num <= m for num in nums)
            if cnt <= m: i = m + 1
            else: j = m
        return i


# 378. 有序矩阵中第K小的元素
class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """

        r, c = len(matrix), len(matrix[0])
        i, j = matrix[0][0], matrix[-1][-1]
        while i < j:
            m = (i + j) >> 1
            cnt = sum(matrix[x][y] <= m for x in range(r) for y in range(c))
            if cnt < k: i = m + 1 # 若<=中值的个数cnt小于k,则第k个值肯定在中值的右侧
            else: j = m
        return i
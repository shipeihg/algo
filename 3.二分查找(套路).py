# 二分总结模板
# 有了这套模板，女朋友再也不用担心我刷不动 LeetCode 了
# http://www.shangdixinxi.com/detail-1104691.html
# https://zhuanlan.zhihu.com/p/86136802

"""

while left < right:
    mid = (left + right) >> 1 或者 mid = (left + right + 1) >> 1，这里有两个选择，是因为取中点时有两种可能：左中位点和右中位点
    if 选择右中位数：
        right = mid - 1

    if 选择左中位数:
        left = mid + 1

    保证选择的中位数，可以让区间收缩

"""

# https://leetcode-cn.com/problems/search-insert-position/solution/te-bie-hao-yong-de-er-fen-cha-fa-fa-mo-ban-python-/


# 153. 寻找旋转排序数组中的最小值
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] > nums[-1]: # 这里-1替换为right是ac通过
                left = mid + 1
            else:
                right = mid
        return nums[left]


# 540. Single Element in a Sorted Array (Medium)
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) >> 1
            if mid % 2 == 1:
                mid -= 1 # 始终保持m为偶数，single之前，必然满足m是偶数&&nums[i]==nums[i+1]
            if nums[mid] == nums[mid + 1]:
                left = mid + 2
            else:
                right = mid
        return nums[left]



# 34. 在排序数组中查找元素的第一个和最后一个位置
# https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/si-lu-hen-jian-dan-xi-jie-fei-mo-gui-de-er-fen-cha/
class Solution:
    def searchLeft(self, a, k):
        i, j = 0, len(a)-1
        while i < j:
            m = (i + j) >> 1
            if a[m] < k: i = m + 1
            else: j = m
        return i if a[i]==k else -1 # 提前判断数组中不存在target的情况
    
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


# 154. 寻找旋转排序数组中的最小值 II
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
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        left = 1
        right = len(nums) - 1 # 在[1, n]范围内寻找目标值
        while left < right:
            mid = (left + right) >> 1
            cnt = sum(num <= mid for num in nums)
            if cnt <= mid:
                left = mid + 1
            else:
                right = mid
        return left


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
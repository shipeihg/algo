
# 15. 三数之和 (对撞指针)
# https://leetcode-cn.com/problems/3sum/solution/hua-jie-suan-fa-15-san-shu-zhi-he-by-guanpengchn/
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums.sort()
        ans = []
        for i in range(len(nums)):
            if nums[i] > 0: return ans
            if i > 0 and nums[i] == nums[i-1]: continue
            left, right = i+1, len(nums)-1
            while left < right:
                s = nums[i] + nums[left] + nums[right]
                if s == 0: 
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]: left += 1 # 当s=0, nums[left] == nums[left+1],应该去重
                    while left < right and nums[right] == nums[right-1]: right -= 1
                    left += 1
                    right -= 1
                elif s < 0: left += 1
                else: right -= 1
        return ans


# 剑指offer: 将数组分为两部分，一半是可以被2整除，一半不可以，空间复杂度O(1)   (对撞指针)  
class Solution:
    def splitArray(self, nums):
        i, j = 0, len(nums)-1
        while i < j:
            while i < len(nums) and nums[i] & 1 == 1: i += 1
            while j >= 0 and nums[j] & 1 == 0: j -= 1
            if i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

# 27. 移除元素(快慢指针原地操作)
#https://leetcode-cn.com/problems/remove-element/solution/python-shuang-zhi-zhen-da-fa-hao-a-quan-guo-zui-ca/
class Solution(object):
    def removeElement(self, nums, val):
        slow = fast = 0
        while fast < len(nums):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1 # slow之前的元素都是满足条件的，即都是不等于val的
            fast += 1
        return slow

# 26 去除数组的重复数 (快慢)
class Solution:
    def removeDupliates(self, arr):
        slow, fast = 0, 1
        while fast < len(arr):
            if arr[slow] != arr[fast]:
                slow += 1
                arr[slow] = arr[fast]
            fast += 1
        return arr[:slow+1]

# 283. 移动零 (快慢指针)
class Solution(object):
    def moveZeroes(self, nums):
        slow = fast = 0 # 快慢指针
        while fast < len(nums):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1 # slow之前的元素都是满足条件的，即都是是非0的
            fast += 1
    
# 分发饼干
class Solution(object):
    def findContentChildren(self, g, s):
        g.sort()
        s.sort()
        i = j = count = 0
        while i<len(g) and j<len(s):
            if s[j] < g[i]: # 饼干小于胃口
                j+=1
            else:
                i+=1
                j+=1
        return i

# 350. 两个数组的交集 II
class Solution(object):
    def intersect(self, a, b):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        a.sort()
        b.sort()
        i = j = 0
        ans = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                ans.append(a[i])
                i += 1
                j += 1
            elif a[i] > b[j]: j += 1
            else: i += 1
        return ans
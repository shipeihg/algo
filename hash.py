

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype:
        """
        
        d = {}
        for i, n in enumerate(nums):
            if target - n in d:
                return [d[target-n], i]
            d[n] = i
        return [-1, -1]


class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        ma = 0
        s = set(nums)
        for n in nums:
            # 这里应该有个 if n-1 not in s,可以提高速度
            curNum = n
            curLen = 1
            while curNum + 1 in s:
                curNum += 1
                curLen += 1
            ma = max(ma, curLen)
        return ma


class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        ma = 0
        s = set(nums)
        for n in nums:
            if n-1 not in s:
                curNum = n
                curLen = 1
                while curNum+1 in s:
                    curNum += 1
                    curLen += 1
                ma = max(ma, curLen)
        return ma
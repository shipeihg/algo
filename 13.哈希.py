

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



# 128. 最长连续序列      
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        curLen = maxLen = 0 
        s = set(nums)
        for n in nums:
            if n-1 not in s:
                curLen = 1
                curNum = 1 + n
                while curNum in s:
                    curNum += 1
                    curLen += 1
            maxLen = max(maxLen, curLen)
        return maxLen


# 560. 和为K的子数组   
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        preSum = {0: 1}; sm = 0; ret = 0
        for n in nums:
            sm += n
            ret += preSum.get(sm-k, 0)
            preSum[sm] = preSum.get(sm, 0) + 1
        return ret


# 670 最大交换(头条问的)

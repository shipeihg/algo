class Solution(object):
    def partitionLabels(self, S):

        last = {c:i for i,c in enumerate(S)}
        
        start = 0; end = last[S[0]]; cnt = 0; r = []
        for i,c in enumerate(S):
            end = max(end, last[c])
            if i == end:
                r.append(end-start+1)
                if i < len(S)-1:
                    start = i+1
                    end = last[S[i+1]]
        return r


class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i+1):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], 1+dp[j])
        return max(dp)
    
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        
        dp = [1 + amount] * (1 + amount)
        dp[0] = 0
        for i in range(1, 1+amount):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], dp[i-c])
        if dp[-1] == 1 + amount: return -1
        return dp[-1]


class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        if not coins: reutrn 0
        for coin in coins:
            for i in range(coin, 1+amount):
                dp[i] += dp[i-coin]
        return dp[-1]


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        
        n = len(s)
        dp = [False] * (1 + n)
        dp[0] = True
        for i in range(1, 1 + n):
            for word in wordDict:
                if len(word) <= i and s[i - len(word): i]:
                    dp[i] = dp[i] or dp[i - len(word)]
        return dp[-1]


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        import numpy as np
        dp = np.zeros((len(prices, 2))).tolist()
        dp[0][0] = 0; dp[0][1] = -prices[0]
        for i in range(0, len(prices)):
            for j in range(2):
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
                dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]



class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp_pre_0, dp_i_0, dp_i_1 = 0, -prices[0] # -prices[0]第一天持有，不可能
        for i in range(1, len(prices)):
            tmp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, dp_pe_0 - prices[i])
            dp_pre_0 = tmp
        return dp_i_0
     
        
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        dp_i_0, dp_i_1 = 0, -float('inf')
        for i in range(len(prices)):
            tmp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, tmp - prices[i] - fee)
        return dp_i_0


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # self.ma = self.cnt = 0
        # def countOnes(n):
        #     if nums[n] == 0:
        #         self.ma = max(self.ma, self.cnt)
        #         self.cnt = 0
        #     if nums[n] == nums[n-1]:
        #         self.cnt += 1
        
        cur = ma = 0
        for n nums:
            cur = 0 if n==0 else 1+cur
            ma = max(ma, cur)
        return ma
                
        

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]: return False
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n-1
        while row < m and col >=0:
            if matrix[row][col] == target: return True
            elif matrix[row][col] < target: row += 1
            else: col -= 1
        return False
    
    
class Solution(object):
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        A = nums
        ma = 0
        for i in range(len(nums)):
            s = set()
            new_index = A[i]
            while new_index not in s:
                s.add(new_index)
                new_index = A[new_index]
            ma = max(ma, len(s))
        return ma
            
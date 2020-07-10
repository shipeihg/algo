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
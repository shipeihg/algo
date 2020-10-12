
# 剑指 Offer 49. 丑数 (DP)
class Solution:
    def GetUglyNumber_Solution(self, n):
        if n == 0: return 0
        dp = [1] * n 
        a = b = c = 0
        for i in range(1, n):
            dp[i] = min(2*dp[a], 3*dp[b], 5*dp[c])
            if dp[i] == 2*dp[a]: a += 1
            if dp[i] == 3*dp[b]: b += 1
            if dp[i] == 5*dp[c]: c += 1
        return dp[-1]

# 只有两个键的键盘（650）
class Solution(object):
    def minSteps(self, n):
        """
        :type n: int
        :rtype: int
        """
        def F(n):
            if n == 1: return 0
            for i in range(int(n/2), 1, -1):
                if n % i == 0: return F(i) + n / i
            return n
        return F(n)
    

#547. 朋友圈 (DFS)
class Solution(object):
    def findCircleNum(self, M):
        def F(i):
            for k,v in enumerate(M[i]):
                if v==1 and k not in circle:
                    circle.add(k)
                    F(k)
                    
        ans = 0; circle = set()
        for i in range(len(M)):
            if i not in circle:
                F(i)
                ans += 1
        return ans

# 56. 合并区间
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if not intervals: return []
        intervals.sort()
        ans = [intervals[0]]
        for i in range(1, len(intervals)):
            if ans[-1][1] < intervals[i][0]: ans.append(intervals[i])
            else: ans[-1][1] = max(ans[-1][1], intervals[i][1])
        return ans


# 折纸问题
def fold(n):
    if n == 1: return '1'
    else: return fold(n-1) + '1' + ''.join(map(lambda c: str(1-int(c)), fold(n-1)))[::-1]
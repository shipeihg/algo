
# 696. 计数二进制子串
class Solution(object):
    cnt = 0 # 全局计数变量
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        def extend(s, i, j, preSI, preSJ):
            if i < 0 or j > len(s)-1: return # 超出边界，返回
            if s[i] == s[j]: return # s[i]=s[j]，不符合规则，返回
            if s[i] == preSI and s[j] == preSJ: # s[i]!=s[j]并且s[i]、s[j]分别和前一个数字保持一致，则记数加以，继续递归
                self.cnt += 1
                extend(s, i-1, j+1, s[i], s[j]) # 将preSI、preSJ分别赋予当前的值，便于下一次递归使用
            
        for i in range(len(s)-1):
            extend(s, i, i+1, s[i], s[i+1])
        return self.cnt


class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        
        if x == 0: return True
        if x < 0 or x % 10 == 0: return False
        xcopy = x
        
        y = 0
        while x > 0:
            y = y * 10 + x % 10
            x /= 10
        return xcopy == y

# 415. 字符串相加    
class Solution:
    def addStrings(self , s , t ):
        # write code here
        
        if s == '0': return t 
        if t == '0': return s 
        
        m = max(len(s), len(t))
        s, t = '0'*(m-len(s)) + s, '0'*(m-len(t)) + t
        ans = [0]*(m+1)
        for i in range(m, 0, -1):
            ans[i] = int(s[i-1]) + int(t[i-1]) # 注意ans数组要比s1,s2长度多1,要错开
        for i in range(m, 0, -1):
            ans[i-1] += ans[i] // 10
            ans[i] %= 10
        ans = ''.join(str(x) for x in ans)
        return ans if ans[0]=='1' else ans[1:]
    
    
# 43. 字符串相乘
# https://leetcode-cn.com/problems/multiply-strings/solution/zi-fu-chuan-xiang-cheng-by-leetcode-solution/
class Solution(object):
    def multiply(self, s1, s2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if s1=='0' or s2=='0': return '0'

        m, n = len(s1), len(s2)
        ans = [0] * (m+n)
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                ans[i+j+1] += int(s1[i]) * int(s2[j])

        for i in range(m+n-1, 0, -1):
            ans[i-1] += ans[i] // 10   # 利用当前的和计算前一位的进位
            ans[i] %= 10 # 当前位取余
        
        start = 1 if ans[0] == 0 else 0
        return ''.join(str(s) for s in ans)[start:]

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
    
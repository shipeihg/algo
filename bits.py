
# 260. 只出现一次的数字 III
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        diff = reduce(lambda x,y: x ^ y, nums)
        diff = diff & (-diff)
        
        a, b = 0, 0
        for n in nums:
            if n & diff == 0: a ^= n
            else: b ^= n
        return [a,b]
        

# 318. 最大单词长度乘积      
class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        
        binary = [0] * len(words)
        for i, word in enumerate(words):
            for c in word:
                binary[i] = binary[i] or (1 << (ord(c) - ord('a')))
        
        ma = 0
        for i in range(len(words)):
            for j in range(i, len(words)):
                if binary[i] & binary[j] == 0:
                    ma = max(ma, len(words[i] * words[j]))
        return ma
    
            
            
        